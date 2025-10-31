import asyncio
import base64
import json
import logging
import random
import io
from collections import defaultdict
from pathlib import Path
import re
from typing import Dict, List, Tuple, Set, Any




import numpy as np
import zendriver as zd
from PIL import Image
from zendriver import cdp

from wplace_bot.analysis.human_placer import get_human_like_ordered_pixels
from wplace_bot.utils.color_helper import find_color_ids_with_alpha
from wplace_bot.config import (
    COLOR_PALETTE, COLOR_PALETTE_JSON,
)


class BrowserInteractionError(Exception):
    """Custom exception for critical browser failures that should terminate the session."""
    pass


def find_closing_brace(text: str, start_index: int) -> int:
    """
    Given a text and the index of an opening brace '{', finds the index of the 
    corresponding closing brace '}'. Returns -1 if not found.
    """
    if text[start_index] != '{':
        return -1
        
    brace_level = 1
    for i in range(start_index + 1, len(text)):
        char = text[i]
        if char == '{':
            brace_level += 1
        elif char == '}':
            brace_level -= 1
            if brace_level == 0:
                return i
    return -1

def expose_paint_function(body_str: str, script_name: str) -> tuple[str, str | None]:
    """
    Dynamically and safely finds the paint function, its class, and its instance,
    then injects code to expose it on the window object.
    
    Args:
        body_str (str): The string containing the minified JavaScript code.
        script_name (str): The name of the script being analyzed, for logging.
        
    Returns:
        tuple[str, str | None]: A tuple containing:
            - The modified (or original) JavaScript code as a string.
            - The found paint function's name, or None if not found.
    """
    # Step 1: Find our anchor - the `paint` function containing the unique token.
    paint_func_regex = re.compile(r'(async\s+paint)\s*\([^)]*\)\s*{[\s\S]*?"x-pawtect-token"')
    paint_match = paint_func_regex.search(body_str)

    if not paint_match:
        return body_str, None

    paint_start_index = paint_match.start()
    paint_func_name_from_match = "paint" # We are matching the literal name 'paint'
    logging.debug(f"[+] Found anchor 'paint' function at index {paint_start_index}.")

    # Step 2: Work backwards to find candidate classes.
    class_def_regex = re.compile(r'class\s+([a-zA-Z0-9_$]+)\s*{')
    preceding_code = body_str[:paint_start_index]
    candidate_classes = list(class_def_regex.finditer(preceding_code))

    if not candidate_classes:
        logging.error("[-] No preceding 'class' definitions found before the anchor function.")
        return body_str, None

    found_class_name = None
    for class_candidate in reversed(candidate_classes):
        class_name = class_candidate.group(1)
        class_open_brace_index = class_candidate.end(0) - 1
        
        # Step 3: Verify Scope.
        class_close_brace_index = find_closing_brace(body_str, class_open_brace_index)
        if class_close_brace_index != -1 and paint_start_index > class_open_brace_index and paint_start_index < class_close_brace_index:
            logging.debug(f"[+] VERIFIED: 'paint' function is inside class '{class_name}'.")
            found_class_name = class_name
            break

    if not found_class_name:
        logging.error("[-] FAILED: Could not verify the parent class for the 'paint' function.")
        return body_str, None

    # Step 4: Find where this confirmed class is instantiated.
    instance_regex = re.compile(
        r"(?:let|var|const)\s+([a-zA-Z0-9_$]+)\s*=\s*new\s+" + re.escape(found_class_name) + r"\s*\([^)]*\);"
    )
    instance_match = instance_regex.search(body_str)

    if not instance_match:
        logging.error(f"[-] Found class '{found_class_name}', but could not find where it was instantiated.")
        return body_str, None

    instance_name = instance_match.group(1)
    full_instantiation_line = instance_match.group(0)
    logging.debug(f"[+] Found class instance variable: '{instance_name}'")

    # Step 5: Inject the binding code safely after the instantiation line.
    injection_code = f"window.exposedPaintFunction = {instance_name}.paint.bind({instance_name});"
    replacement_block = f"{full_instantiation_line}\n{injection_code}"
    
    modified_body_str = body_str.replace(full_instantiation_line, replacement_block, 1)

    logging.info(f"[+] Exposed paint function '{instance_name}.paint' as 'window.exposedPaintFunction' from '{script_name}'")
    
    return modified_body_str, paint_func_name_from_match

class WplacePainter:
    """
    Manages browser automation, user state, and network interactions with wplace.live.
    """
    def __init__(self, pixel_queue: List[Tuple[float, int, int, int]], transparent_pixels: List[Tuple[int, int]], burn_candidates: List[List[Tuple[int, int]]]):
        """
        Initializes the painter with pre-analyzed pixel data.

        Args:
            pixel_queue: A list of (priority, gx, gy, color_id) tuples to paint.
            transparent_pixels: A list of (gx, gy) tuples representing available transparent pixels.
            burn_candidates: A list of lists, where each inner list contains (gx, gy)
                             tuples from a specific burn strategy, in execution order.
        """
        self.browser: zd.Browser | None = None
        self.page: zd.Tab | None = None
        self.user_data: Dict[str, Any] | None = None
        self.current_charges: int = 0
        self.max_charges: int = 0
        self.droplets: int = 0
        self.available_colors: Set[int] = set()
        self.fingerprint_func_name: str | None = None


        # --- Data from external analysis ---
        self.pixel_queue = pixel_queue
        self.available_transparent_pixels = transparent_pixels
        self.burn_candidates = burn_candidates

    async def setup_browser(self, profile_dir: Path) -> tuple[str | None, str | None, str | None]:
        """
        Starts and configures a browser instance with JS interception.
        The found function names are returned by this method.
        """
        logging.info(f"Setting up browser for profile: {profile_dir.name}...")
        
        fingerprint_func_name: str | None = None
        paint_func_name: str | None = None
        fingerprint_func_found = asyncio.Future()
        paint_func_found = asyncio.Future()
        
        config = zd.Config(headless=False, user_data_dir=str(profile_dir.absolute()), browser="brave",

                        )
        self.browser = await zd.start(config)
        self.page = self.browser.main_tab
        for tab in self.browser.tabs:
            if tab != self.page:
                await tab.close()
        active_requests_count = 0

        async def on_request_paused(event: cdp.fetch.RequestPaused):
            nonlocal fingerprint_func_name, paint_func_name, active_requests_count
            active_requests_count += 1
            request_id = event.request_id
            try:
                script_name = event.request.url.split('/')[-1].split('?')[0]
                if event.response_status_code is None or event.resource_type != cdp.network.ResourceType.SCRIPT:
                    await self.page.send(cdp.fetch.continue_request(request_id=request_id))
                    return

                body_content, is_base64 = await self.page.send(
                    cdp.fetch.get_response_body(request_id=request_id)
                )
                body_str = base64.b64decode(body_content).decode('utf-8', 'replace') if is_base64 else body_content
                modified_body_str = body_str

                # --- Find Fingerprint function (unchanged) ---
                if not fingerprint_func_found.done():
                    fp_match = re.search(r'async function\s+([a-zA-Z0-9_$]+)\(\)\s*\{\s*return\s*\(await\s*\(await\s*.*\)\.get\(\)\)\.visitorId', body_str)
                    if fp_match:
                        fingerprint_func_name = fp_match.group(1)
                        modified_body_str += f"\n;window.getFingerprint = {fingerprint_func_name};\n"
                        logging.info(f"[+] Found fingerprint function '{fingerprint_func_name}'")
                        logging.info(f"[+] Exposed fingerprint function as 'window.getFingerprint' in '{script_name}'")
                        fingerprint_func_found.set_result(True)
                
                # --- MODIFIED BLOCK: Use the new robust function for paint ---
                if not paint_func_found.done():
                    # The new function handles both finding and modifying the body
                    modified_body_str, found_name = expose_paint_function(modified_body_str, script_name)
                    if found_name:
                        paint_func_name = found_name
                        paint_func_found.set_result(True)
                # --- END OF MODIFIED BLOCK ---

                encoded_body = base64.b64encode(modified_body_str.encode('utf-8')).decode('ascii')
                await self.page.send(cdp.fetch.fulfill_request(
                    request_id=request_id,
                    response_code=event.response_status_code,
                    response_headers=event.response_headers or [],
                    body=encoded_body
                ))
            except Exception as e:
                logging.error(f"Error in request handler for {event.request.url}: {e}")
                if not self.page.closed:
                    try:
                        await self.page.send(cdp.fetch.continue_request(request_id=request_id))
                    except Exception as final_e:
                        logging.error(f"Failed to continue request {request_id} after error: {final_e}")
            finally:
                active_requests_count -= 1

        self.page.add_handler(cdp.fetch.RequestPaused, on_request_paused)
        await self.page.send(cdp.network.set_cache_disabled(cache_disabled=True))
        await self.page.send(cdp.fetch.enable(patterns=[
            cdp.fetch.RequestPattern(resource_type=cdp.network.ResourceType.SCRIPT, request_stage=cdp.fetch.RequestStage.RESPONSE)
        ]))

        await self.page.get("https://wplace.live/")
        await self.page.reload()

        logging.info("Waiting for helper functions to be exposed on window object...")
        try:
            await asyncio.wait_for(
                asyncio.gather(fingerprint_func_found,  paint_func_found),
                timeout=30.0
            )
            logging.info("All three helper functions found and exposed.")
        except asyncio.TimeoutError:
            raise Exception("Cannot run without finding all helper functions")
        
        logging.info("Waiting for page to complete loading...")
        await self.page.wait_for_ready_state('complete', timeout=30)

        while active_requests_count > 0:
            logging.info(f"Waiting for {active_requests_count} in-flight request(s) to complete...")
            await asyncio.sleep(0.1)

        self.page.remove_handlers(cdp.fetch.RequestPaused, on_request_paused)
        await self.page.send(cdp.fetch.disable())
        logging.info("Request interception handler removed")

        return fingerprint_func_name, paint_func_name

    async def _fetch_and_update_user_data(self):
        """Fetches the /me endpoint to update the user's state."""
        if not self.page:
            return

        logging.info("Fetching latest user data from /me endpoint...")
        js_code = """  
        (async () => {  
            const r = await fetch("https://backend.wplace.live/me", {credentials: 'include'});  
            const data = await r.json();  
            return JSON.stringify(data);  // Return as string  
        })();  
        """  

        try:
            new_user_data_str = await self.page.evaluate(js_code, await_promise=True)
            if isinstance(new_user_data_str, tuple): # Handle zendriver's complex return types
                new_user_data_str = new_user_data_str[0].value if new_user_data_str[0] else None
            elif hasattr(new_user_data_str, 'value'):
                new_user_data_str = new_user_data_str.value
            
            new_user_data = json.loads(new_user_data_str)
            if new_user_data and 'id' in new_user_data:
                self.user_data = new_user_data
                self.update_state_from_user_data()
            else:
                logging.warning(f"Failed to fetch user data, received: {new_user_data}")
        except Exception as e:
            logging.error(f"Exception while fetching user data: {e}")
            raise BrowserInteractionError("Failed to fetch and update user data.") from e

    def update_state_from_user_data(self):
        """Parses the user data dictionary to update internal state."""
        if not self.user_data: return
        self.current_charges = int(self.user_data.get('charges', {}).get('count', 0))
        self.max_charges = int(self.user_data.get('charges', {}).get('max', 0))
        self.droplets = int(self.user_data.get('droplets', 0))
        logging.info(f"State updated: {self.current_charges}/{self.max_charges} charges, {self.droplets} droplets.")
        if 'extraColorsBitmap' in self.user_data:
            self._parse_available_colors(int(self.user_data['extraColorsBitmap']))

    def _parse_available_colors(self, bitmap: int):
        self.available_colors = set(range(32)) # Base colors
        self.available_colors.add(0) # Transparent
        if bitmap < 0: bitmap += (1 << 32)
        binary_bitmap_reversed = bin(bitmap)[2:].zfill(32)[::-1]
        for i, bit in enumerate(binary_bitmap_reversed):
            if bit == '1' and (32 + i) < len(COLOR_PALETTE):
                self.available_colors.add(32 + i)
        logging.info(f"Parsed available colors: {len(self.available_colors)} total.")
        
    async def _filter_and_select_pixels_to_paint(self) -> List[Tuple[int, int, int]]:
        """
        Uses all available charges to paint template pixels, then burns any
        remaining charges strategically.
        """
        # --- Template Painting ---
        # We will always use all available charges.
        spendable_charges = self.current_charges
        if spendable_charges == 0:
            logging.info("No charges available to paint.")
            return []

        logging.info(f"Have {spendable_charges} charges to spend.")

        # Filter the queue for colors the user owns.
        candidate_pixels = [(p, gx, gy, cid) for p, gx, gy, cid in self.pixel_queue if cid in self.available_colors]

        final_pixel_payload = []
        if not candidate_pixels:
            logging.info("No template pixels need painting with available colors.")
        else:
            logging.info(f"Found {len(candidate_pixels)} paintable template pixels. Selecting up to {spendable_charges}.")
            # Use the human-like placer to select the most important pixels up to our charge limit.
            final_pixel_payload = get_human_like_ordered_pixels(candidate_pixels, spendable_charges)
            logging.info(f"Selected {len(final_pixel_payload)} template pixels to paint.")

        # --- Charge Burning Logic ---
        # Calculate remaining charges and burn all of them sequentially by strategy.
        charges_left_to_burn = self.current_charges - len(final_pixel_payload)
        
        if charges_left_to_burn > 0:
            logging.info(f"Will burn {charges_left_to_burn} remaining charges.")
            
            # --- STRATEGIC CHARGE BURNING (SEQUENTIAL) ---
            if self.burn_candidates:
                logging.info(f"Executing {len(self.burn_candidates)} burn strategies in order.")
                for i, strategy_candidates in enumerate(self.burn_candidates):
                    if charges_left_to_burn == 0:
                        break # Stop if we've used all charges

                    # Shuffle candidates within a single strategy to avoid painting in a straight line
                    random.shuffle(strategy_candidates)
                    
                    num_to_take = min(charges_left_to_burn, len(strategy_candidates))
                    
                    if num_to_take > 0:
                        pixels_from_this_strategy = strategy_candidates[:num_to_take]
                        for gx, gy in pixels_from_this_strategy:
                            final_pixel_payload.append((gx, gy, 0)) # Color 0 is Transparent
                        
                        charges_left_to_burn -= num_to_take
                        logging.info(f"Added {num_to_take} pixels from burn strategy #{i+1}.")

            # Fallback if strategic burns didn't use up all charges or weren't available
            if charges_left_to_burn > 0 and self.available_transparent_pixels:
                logging.info(f"Falling back to random transparent strategy for {charges_left_to_burn} remaining charge(s).")
                
                # Group transparent pixels by tile to place them in a concentrated area.
                pixels_by_tile = defaultdict(list)
                for gx, gy in self.available_transparent_pixels:
                    pixels_by_tile[(gx // 1000, gy // 1000)].append((gx, gy))

                if pixels_by_tile:
                    # Find the tile with the most available transparent pixels for burning
                    best_tile = max(pixels_by_tile, key=lambda k: len(pixels_by_tile[k]))
                    fallback_burn_candidates = pixels_by_tile[best_tile]
                    
                    num_added = min(charges_left_to_burn, len(fallback_burn_candidates))
                    coords_to_paint_fallback = random.sample(fallback_burn_candidates, num_added)
                    for gx, gy in coords_to_paint_fallback:
                        final_pixel_payload.append((gx, gy, 0))
                    logging.info(f"Added {num_added} transparent pixels to burn on tile {best_tile} (fallback).")

        return final_pixel_payload
        
    async def _send_authenticated_post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any] | None:
        """Helper to send a generic authenticated POST request via browser fetch."""
        if not self.page:
            logging.error("Cannot send POST: page is not available.")
            return None

        # Define necessary headers for the POST request locally.
        # Authentication headers are handled automatically by `credentials: 'include'`.
        headers = {'Content-Type': 'text/plain;charset=UTF-8'}
        payload_json = json.dumps(payload, separators=(',', ':'))

        js_code = f"""
        (async () => {{
            const r = await fetch({json.dumps(url)}, {{
                method: 'POST',
                headers: {json.dumps(headers)},
                body: {json.dumps(payload_json)},
                credentials: 'include'
            }});
            return {{ ok: r.ok, status: r.status, text: await r.text() }};
        }})();
        """
        try:
            raw_result = await self.page.evaluate(js_code, await_promise=True)
            result = raw_result
            if isinstance(raw_result, tuple):
                result = raw_result[0].value if raw_result[0] else None
            elif hasattr(raw_result, 'value'):
                result = raw_result.value
            return result
        except Exception as e:
            logging.error(f"Failed to execute authenticated POST to {url}: {e}")
            return None

    async def _handle_purchases(self) -> bool:
        """
        Checks for and executes purchases for colors or max charges based on priority rules.
        """
        missing_colors = set(range(len(COLOR_PALETTE))) - self.available_colors
        
        # --- Priority 1: Buy Missing Colors ---
        # If there are missing colors and we have enough droplets, buy one.
        if missing_colors and self.droplets >= 2000:
            color_to_buy = random.choice(list(missing_colors))
            color_name = COLOR_PALETTE_JSON[color_to_buy]['name']
            logging.info(f"Attempting to buy missing color '{color_name}' (ID: {color_to_buy}).")
            payload = {"product": {"id": 100, "amount": 1, "variant": color_to_buy}}
            result = await self._send_authenticated_post("https://backend.wplace.live/purchase", payload)
            
            if result and result.get("ok"):
                logging.info(f"Purchase request for color '{color_name}' sent successfully.")
            else:
                logging.warning(f"Purchase for color failed. Response: {result}")
            return True  # A purchase was attempted

        # --- Priority 2: Buy Max Charges ---
        droplets_available_for_charges = 0
        
        # Condition A: Always buy if max charges are below 200 and we can afford it.
        if self.max_charges < 200 and self.droplets >= 500:
            droplets_available_for_charges = self.droplets
            logging.info(f"Max charges ({self.max_charges}) are below 200. Using all available droplets for upgrades.")
        
        # Condition B: Spend any droplets over the 20,000 threshold.
        elif self.droplets > 20000:
            droplets_available_for_charges = self.droplets - 20000
            logging.info(f"Droplets ({self.droplets}) are over 20,000. Using excess for upgrades.")

        # If either condition provided droplets for charges, execute the purchase.
        if droplets_available_for_charges >= 500:
            amount_to_buy = droplets_available_for_charges // 500
            if amount_to_buy > 0:
                logging.info(f"Attempting to buy {amount_to_buy} max charge upgrades.")
                payload = {"product": {"id": 70, "amount": amount_to_buy}}
                result = await self._send_authenticated_post("https://backend.wplace.live/purchase", payload)
                
                if result and result.get("ok"):
                    logging.info(f"Purchase request for max charges sent successfully.")
                else:
                    logging.warning(f"Purchase for max charges failed. Response: {result}")
                return True  # A purchase was attempted
            
        return False # No purchase was attempted


    async def _execute_paint_requests(self, pixels_to_paint: List[Tuple[int, int, int]]) -> int:
        """
        Retrieves the fingerprint with a retry mechanism, simulates user interaction,
        and then uses the exposed JS function to paint the actual batch of pixels.
        """
        if not self.page or not pixels_to_paint:
            return 0

        # --- Step 1: Get fingerprint with robust retry logic ---
                # --- Step 1: Get fingerprint with robust retry logic ---
        fp = None
        # Increased timeout to 20 seconds total, which is more realistic for fingerprinting
        max_retries = 40
        retry_interval = 1
        js_fp_expr = "(async () => await window.getFingerprint())()"

        logging.info("Attempting to retrieve fingerprint with robust polling...")
        for attempt in range(max_retries):
            try:
                raw_fp_obj = await self.page.evaluate(js_fp_expr, await_promise=True)
                
                # Use the helper to safely get the value
                candidate_fp = raw_fp_obj

                # --- CRITICAL DEBUGGING STEP ---
                # Log what we receive on each attempt to see the progress.
                logging.debug(f"Fingerprint retrieval attempt {attempt + 1}: Received '{str(candidate_fp)[:100]}'")

                if isinstance(candidate_fp, str) and candidate_fp:
                    fp = candidate_fp
                    logging.info(f"Successfully retrieved fingerprint after {attempt + 1} attempt(s): {fp[:20]}...")
                    break  # Success, exit the loop
            
            except Exception as e:
                # This catches errors in the JS execution itself
                logging.warning(f"Exception during fingerprint retrieval attempt {attempt + 1}: {e}")

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_interval)
        
        if not fp:
            logging.error("Failed to retrieve fingerprint after all retries. The function may have returned null/undefined.")
            raise BrowserInteractionError("Failed to retrieve fingerprint after multiple attempts.")
        
        # --- Step 2: Simulate a UI paint and block the request to avoid wasting a charge ---
        ui_paint_blocked_future: asyncio.Future[bool] = asyncio.Future()

        async def on_paint_request_paused(event: cdp.fetch.RequestPaused):
            if "backend.wplace.live/s0/pixel/" in event.request.url and event.request.method == 'POST':
                if not ui_paint_blocked_future.done():
                    try:
                        await self.page.send(cdp.fetch.fail_request(
                            request_id=event.request_id,
                            error_reason=cdp.network.ErrorReason.ABORTED
                        ))
                        ui_paint_blocked_future.set_result(True)
                    except Exception as e:
                        if not ui_paint_blocked_future.done(): ui_paint_blocked_future.set_exception(e)
                return

            try:
                if not self.page.closed: await self.page.send(cdp.fetch.continue_request(request_id=event.request_id))
            except Exception: pass

        await self.page.send(cdp.fetch.enable())
        self.page.add_handler(cdp.fetch.RequestPaused, on_paint_request_paused)

        try:
            logging.info("Simulating UI paint action to appear more human...")
            await (await self.page.select(r'div[class*="bottom-3"] button.btn-primary:not(:disabled)')).click()
            await (await self.page.select("#color-0")).click()
            await (await self.page.select("canvas.maplibregl-canvas")).click()
            paint_button_selector = 'div[class*="left-1/2"] button.btn-primary:not(:disabled)'
            logging.info(f"Waiting for final paint button to become enabled...")
            paint_button = await self.page.select(paint_button_selector, timeout=15)
            await paint_button.click()
            await asyncio.wait_for(ui_paint_blocked_future, timeout=15)
            logging.info("UI paint request successfully intercepted and blocked.")
        except Exception as e:
            logging.error(f"Error during simulated UI interaction: {e}", exc_info=True)
            raise BrowserInteractionError("Failed to simulate UI paint click.") from e
        finally:
            self.page.remove_handlers(cdp.fetch.RequestPaused, on_paint_request_paused)
            if not self.page.closed: await self.page.send(cdp.fetch.disable())
            logging.info("Network interception for UI simulation disabled.")

        # --- Step 3: Prepare data for the real paint call ---
        pixel_js_array = []
        season = 0
        for gx, gy, color_id in pixels_to_paint:
            tx, px = divmod(gx, 1000)
            ty, py = divmod(gy, 1000)
            pixel_js_array.append({
                "tile": [int(tx), int(ty)], "season": season,
                "colorIdx": int(color_id), "pixel": [int(px), int(py)]
            })

        # --- Step 4: Call the exposed paint function with the now-valid fingerprint ---
        successful_pixels = 0
        try:
            pixel_data_json = json.dumps(pixel_js_array)
            js_code = f"""
            (async () => {{
                try {{
                    const pixels = {pixel_data_json};
                    const fingerprint = {json.dumps(fp)};
                    await window.exposedPaintFunction(pixels, fingerprint);
                    return {{ "ok": true, "count": {len(pixels_to_paint)} }};
                }} catch (e) {{
                    return {{ "ok": false, "error": e.stack || e.toString() }};
                }}
            }})();
            """
            result_obj = await self.page.evaluate(js_code, await_promise=True)
            result = None
            if isinstance(result_obj, tuple) and result_obj and hasattr(result_obj[0], 'value'):
                result = result_obj[0].value
            elif hasattr(result_obj, 'value'):
                result = result_obj.value
            else:
                result = result_obj

            if result and result.get("ok"):
                successful_pixels = result.get("count", 0)
                logging.info(f"Successfully painted {successful_pixels} pixels via exposed function.")
            else:
                logging.error(f"Failed to paint pixels via exposed function. {result}")
                error_msg = result.get('error') if result else "Unknown error"
                logging.error(f"Paint via exposed function failed with JS error: {error_msg}")
        except Exception as e:
            logging.error(f"Failed to execute paint via exposed function: {e}", exc_info=True)
        
        logging.info(f"Finished painting. Successfully painted {successful_pixels} pixels.")
        return successful_pixels
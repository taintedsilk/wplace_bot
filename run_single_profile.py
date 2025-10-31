# run_single_profile.py
import argparse
import asyncio
import logging
from pathlib import Path
from typing import List

from wplace_bot.analysis.canvas_analyzer import CanvasAnalyzer
from wplace_bot.config import LOG_FILE
from wplace_bot.painter import WplacePainter, BrowserInteractionError
from wplace_bot.utils.file_io import save_profile_data

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode='a'), logging.StreamHandler()]
)

async def process_profile(args):
    """Handles the entire lifecycle for a single profile."""
    profile_name = args.profile_name
    templates_to_analyze: List[Path] = [Path(p) for p in args.target_templates]
    painter = None
    try:
        run_description = f"on {len(templates_to_analyze)} template(s)" if templates_to_analyze else "for a charge-burn run"
        logging.info(f"--- Starting process for profile: {profile_name} {run_description} ---")
        
        profile_dir = Path(args.profiles_folder) / profile_name
        if not profile_dir.is_dir():
            logging.error(f"Profile directory not found: {profile_dir}. Aborting.")
            return

        for t in templates_to_analyze:
            if not t.is_file():
                logging.error(f"Target template file not found: {t}. Aborting.")
                return

        # 1. Analyze all target templates and burn candidates before starting the browser.
        analyzer = CanvasAnalyzer()
        analyzer.analyze_templates(templates_to_analyze)
        analyzer.analyze_burn_candidates()
        
        if not analyzer.pixel_queue:
            if templates_to_analyze:
                logging.info("Re-analysis shows no pixels need fixing across target templates. Will proceed to burn charges if any.")
            else:
                logging.info("No target templates specified. Proceeding with charge burning only.")
            
        # 2. Instantiate painter and start the browser.
        painter = WplacePainter(
            analyzer.pixel_queue, 
            analyzer.available_transparent_pixels,
            analyzer.burn_candidates
        )
        await painter.setup_browser(profile_dir)
        await painter._fetch_and_update_user_data() 

        # 3. Select pixels, paint, and handle purchases.
        pixels_to_paint = await painter._filter_and_select_pixels_to_paint()
        painted_count = 0
        if pixels_to_paint:
            painted_count = await painter._execute_paint_requests(pixels_to_paint)
            logging.info(f"Painted {painted_count} pixels for '{profile_name}'.")
        else:
            logging.info("No pixels were selected to be painted.")

        # 4. Handle purchases after painting.
        if await painter._handle_purchases():
            logging.info("A purchase was attempted. Refreshing user state.")
            await asyncio.sleep(2) # Give the backend a moment to process the purchase
            await painter._fetch_and_update_user_data()

    except (Exception, BrowserInteractionError) as e:
        logging.critical(f"A critical error occurred for profile '{profile_name}': {e}", exc_info=True)
    finally:
        if painter:
            try:
                if painter.page and not painter.page.closed:
                    await painter._fetch_and_update_user_data()
            except Exception as fetch_e:
                logging.error(f"Could not fetch final user data before exit: {fetch_e}")

            if painter.user_data:
                save_profile_data(profile_name, painter.current_charges, painter.max_charges)

            if painter.browser:
                try:
                    if painter.browser.main_tab and not painter.browser.main_tab.closed:
                        await painter.browser.main_tab.save_screenshot(f"{profile_name}_final.png")
                except Exception as ss_e:
                    logging.warning(f"Could not save final screenshot: {ss_e}")
                
                await painter.browser.stop()
        logging.info(f"--- Finished process for profile: {profile_name} ---")

def main():
    parser = argparse.ArgumentParser(description="Run Wplace Painter for a single profile to fix specific templates.")
    parser.add_argument("profile_name", type=str, help="The name of the profile folder to use.")
    parser.add_argument("target_templates", type=str, nargs='*', help="A list of full paths to the .png templates to fix.")
    parser.add_argument("--profiles_folder", type=str, default="./Profiles", help="Folder containing Chrome profiles.")
    args = parser.parse_args()
    
    try:
        asyncio.run(process_profile(args))
    except KeyboardInterrupt:
        logging.info("Script interrupted by user.")

if __name__ == "__main__":
    main()
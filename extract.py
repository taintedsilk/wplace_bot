import argparse 
import os 

def is_human_readable(filepath): 
    """ 
    Determines if a file is human-readable by attempting to decode it using UTF-8. 
    This is a common way to distinguish text files from binary files. 
    """ 
    try: 
        with open(filepath, 'r', encoding='utf-8') as f: 
            f.read(1024)  # Read the first 1KB to test 
    except (UnicodeDecodeError, IOError): 
        return False 
    return True 

def extract_readable_files(directory, output_file): 
    """ 
    Iterates through a directory, identifies human-readable files, 
    and merges their content into a single output file. 
    """ 
    if not os.path.isdir(directory): 
        print(f"Error: The specified directory does not exist: {directory}") 
        return 

    # Get the absolute path of the output file to prevent it from being included. 
    abs_output_file = os.path.abspath(output_file) 
    
    # Get the absolute path of the script itself to prevent it from being included. 
    abs_script_path = os.path.realpath(__file__) 

    try: 
        with open(output_file, 'w', encoding='utf-8') as outfile: 
            # os.walk will also go through subdirectories. For just the top-level, os.listdir could be used. 
            for dirpath, _, filenames in os.walk(directory): 
                for filename in filenames: 
                    filepath = os.path.join(dirpath, filename) 
                    abs_filepath = os.path.abspath(filepath) 

                    # Skip the output file and the script itself. 
                    if abs_filepath == abs_output_file or abs_filepath == abs_script_path: 
                        continue 

                    if is_human_readable(filepath): 
                        print(f"Adding content from: {filepath}") 
                        outfile.write(f"--- Start of content from {filepath} ---\n\n") 
                        try: 
                            with open(filepath, 'r', encoding='utf-8') as infile: 
                                outfile.write(infile.read()) 
                            outfile.write(f"\n\n--- End of content from {filepath} ---\n\n") 
                        except Exception as e: 
                            print(f"Could not read file {filepath}: {e}") 

        print(f"\nSuccessfully extracted all readable files into '{output_file}'") 

    except IOError as e: 
        print(f"Error writing to output file {output_file}: {e}") 
    except Exception as e: 
        print(f"An unexpected error occurred: {e}") 

if __name__ == '__main__': 
    # Initialize the argument parser with a description of the script's purpose. 
    parser = argparse.ArgumentParser( 
        description="A script to extract all human-readable files from a directory and its subdirectories into a single text file.", 
        formatter_class=argparse.RawTextHelpFormatter, 
        epilog="Example usage:\n" 
               "python extract_text.py ./my_code_project combined_source.txt" 
    ) 

    # Add the command-line arguments. 
    parser.add_argument("directory", 
                        help="The path to the input directory to scan for readable files.") 
    parser.add_argument("output_file", 
                        help="The path to the single output text file.") 

    # Parse the arguments provided by the user. 
    args = parser.parse_args() 

    # Call the main function with the parsed arguments. 
    extract_readable_files(args.directory, args.output_file) 
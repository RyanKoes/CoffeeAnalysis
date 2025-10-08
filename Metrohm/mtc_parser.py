import pandas as pd
import xml.etree.ElementTree as ET
import re


def parse_metrohm_mtc(file_path):
    """
    Parses a Metrohm .mtc (XML) file into a pandas DataFrame.

    This function is designed to handle Metrohm's common XML structure,
    where measurement data is often stored as a large, comma-separated
    string of interleaved potential and current values (e.g., [X1, Y1, X2, Y2...])
    within a <curve> element.

    Args:
        file_path (str): The path to the .mtc file.

    Returns:
        pandas.DataFrame: A DataFrame containing the parsed data (Potential, Current),
                          or an empty DataFrame if parsing fails.
    """
    print(f"Attempting to parse file: {file_path}")
    try:
        # Load the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        data_frames = []

        # Find all 'curve' elements, as this is where the measurement data resides
        curves = root.findall('.//curve')

        if not curves:
            print("Error: Could not find any <curve> elements in the file. Check the file structure.")
            return pd.DataFrame()

        for curve in curves:
            # Extract the curve name for identification
            curve_name = curve.find('name').text if curve.find('name') is not None else 'Unnamed Curve'
            print(f"--- Processing Curve: {curve_name} ---")

            raw_data_string = None

            # Strategy: Search for the child tag with the longest text content,
            # which is likely the large, raw data block (potential/current points).
            for child in curve.iter():
                if child.text and len(child.text.strip()) > 50:
                    # Check if the content is primarily numeric (digits, decimals, signs, e/E for scientific notation, commas)
                    # We also allow for common XML line breaks/spaces to be present.
                    if re.match(r'^[\s\d.,\-eE]+$',
                                child.text.strip().replace('\n', '').replace('\r', '').replace(' ', '')):
                        raw_data_string = child.text.strip()
                        break

            if not raw_data_string:
                print(f"Warning: Could not find a large, numeric raw data string in curve '{curve_name}'. Skipping.")
                continue

            # 1. Clean and split the string data
            # Remove all whitespace and split by comma
            values_str = re.sub(r'\s+', '', raw_data_string)
            raw_values = values_str.split(',')

            # 2. Filter and convert to numeric values
            # Filters out potential status flags ('false', 'true') or empty strings
            numeric_values = []
            for val in raw_values:
                # Regex to check for valid floating-point numbers
                if re.match(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?$', val):
                    try:
                        numeric_values.append(float(val))
                    except ValueError:
                        # Should not happen if regex is correct, but safe check
                        continue

            # 3. Reshape the data (assuming interleaved [Potential, Current])
            # For most electrochemistry (CV) data, there are 2 channels.
            if len(numeric_values) < 2:
                print(f"Error: Less than two numeric values found for curve '{curve_name}'. Skipping.")
                continue

            if len(numeric_values) % 2 != 0:
                print(
                    f"Warning: Numeric data count ({len(numeric_values)}) is odd for curve '{curve_name}'. Assuming 2 channels (Potential, Current) and dropping the last point.")
                numeric_values = numeric_values[:-1]

            # Reshape the 1D list into a 2D array (N rows, 2 columns)
            data_2d = [
                (numeric_values[i], numeric_values[i + 1])
                for i in range(0, len(numeric_values), 2)
            ]

            # 4. Create DataFrame
            df = pd.DataFrame(data_2d, columns=['Potential (V)', 'Current (A)'])
            df.insert(0, 'Curve Name', curve_name)
            data_frames.append(df)
            print(f"Successfully parsed {len(df)} data points for {curve_name}.")

        if data_frames:
            # Concatenate all curve data into a single DataFrame
            final_df = pd.concat(data_frames, ignore_index=True)
            print("\nParsing complete. DataFrame Head:")
            print(final_df.head())
            return final_df
        else:
            print("No valid curve data could be extracted.")
            return pd.DataFrame()

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'. Please ensure the file is in the correct directory.")
        return pd.DataFrame()
    except ET.ParseError as e:
        print(
            f"Error: Failed to parse XML from file '{file_path}'. The file might be corrupted or not valid XML. Details: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during parsing: {e}")
        return pd.DataFrame()


# --- Execution Block ---
if __name__ == '__main__':
    # The uploaded file name is 'MetrohmCoffee1.mtc'
    file_path = 'MetrohmCoffee1.mtc'

    # Run the parser
    df_data = parse_metrohm_mtc(file_path)

    if not df_data.empty:
        print("\n--- Final DataFrame Info ---")
        df_data.info()

        # Optional: Save the data to a CSV file
        df_data.to_csv('parsed_metrohm_data.csv', index=False)
        print("\nData also saved to 'parsed_metrohm_data.csv'")

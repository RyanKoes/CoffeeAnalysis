import xml.etree.ElementTree as ET
import csv
import os

def parse_cv_xml_to_csv(xml_file, csv_file):
    """
    Parse electrochemistry data file (.xml or .mtc) and extract potential and current data to CSV.

    Parameters:
    xml_file (str): Path to input file (.xml or .mtc format)
    csv_file (str): Path to output CSV file
    """
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find the potential and i1 (current) elements
    potential_elem = root.find('.//potential')
    current_elem = root.find('.//i1')

    if potential_elem is None or current_elem is None:
        raise ValueError("Could not find potential or i1 data in XML file")

    # The data is comma-separated with line breaks (&#13;)
    potential_text = potential_elem.text.replace('&#13;', '').replace('\n', '')
    current_text = current_elem.text.replace('&#13;', '').replace('\n', '')

    potentials = [float(x.strip()) for x in potential_text.split(',') if x.strip()]
    currents = [float(x.strip()) for x in current_text.split(',') if x.strip()]

    # Verify data lengths match
    if len(potentials) != len(currents):
        print(f"Warning: Data length mismatch - {len(potentials)} potentials vs {len(currents)} currents")

    # Write to CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Potential (V)', 'Current (µA)'])

        for pot, curr in zip(potentials, currents):
            writer.writerow([pot, curr])

    print(f"Successfully converted {len(potentials)} data points to {csv_file}")
    return len(potentials)

if __name__ == "__main__":
    input_directory = "MTC Files"
    output_directory = "Parsed CSV"

    # Loop through files in the input directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for file in os.listdir(input_directory):
        if file.endswith(".mtc"):
            input_file = os.path.join(input_directory, file)
            output_file = os.path.join(output_directory, os.path.splitext(file)[0] + ".csv")
            try:
                num_points = parse_cv_xml_to_csv(input_file, output_file)
                print(f"Converted {file}: {num_points} data points exported to {output_file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")
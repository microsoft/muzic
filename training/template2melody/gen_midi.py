"""Generate midi files from xml files."""

import pathlib
import music21

cwd = pathlib.Path('.')

xml_data_path = cwd / 'training' / 'template2melody' / 'data_xml'
xml_file_list = list(xml_data_path.glob('*.xml'))

midi_data_path = cwd / 'training' / 'template2melody' / 'data_midi'
if not midi_data_path.exists():
    midi_data_path.mkdir()

# Iterate through xml files and generate midi files
for xml_file in xml_file_list:
    try:
        c = music21.converter.parse(xml_file)
        c.write("midi", midi_data_path / xml_file.with_suffix('.mid').name)
    except:
        # Skip erroneous xml file
        print(f"An error occur when converting file {xml_file}")
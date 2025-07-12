import json
import re

def replace_journal_names(bibtex_entry, abbreviation_dict, abbreviation_dict_lower):
    """
    Replace full journal names with their abbreviations in a BibTeX entry.

    Args:
    - bibtex_entry (str): A single BibTeX entry in string format.
    - abbreviation_dict (dict): Dictionary mapping full journal names to abbreviations.

    Returns:
    - str: The modified BibTeX entry with journal names replaced.
    """
    # Regex pattern to find journal names in the BibTeX entry
    pattern = r'journal\s*=\s*{(.*?)}'

    def replace_match(match):
        journal_name = match.group(1)
        journal_name = journal_name.replace("The", "")
        journal_name = journal_name.strip()
        if journal_name in abbreviation_dict:
            return f'journal = {{ {abbreviation_dict[journal_name]} }}'
        if journal_name.lower() in abbreviation_dict_lower:
            v = abbreviation_dict_lower[journal_name.lower()]
            return f'journal = {{ {abbreviation_dict[v]} }}'

        return match.group(0)

    # Use regex to replace full journal names with abbreviated ones
    return re.sub(pattern, replace_match, bibtex_entry)

def process_bibtex_file(input_file, output_file, abbreviation_dict):
    """
    Process a BibTeX file and replace full journal names with their abbreviations.

    Args:
    - input_file (str): Path to the input .bib file.
    - output_file (str): Path to the output .bib file.
    - abbreviation_dict (dict): Dictionary mapping full journal names to abbreviations.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        bibtex_data = file.read()

    abbreviation_dict_lower = {}
    for n in abbreviation_dict:
        abbreviation_dict_lower[n.lower()] = n

    # Replace journal names
    modified_bibtex_data = replace_journal_names(bibtex_data, abbreviation_dict, abbreviation_dict_lower)

    # Write the modified BibTeX data to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(modified_bibtex_data)

# Example usage:
input_bib_file = 'ref.bib'  # Replace with your BibTeX file path
output_bib_file = 'output.bib'  # Replace with desired output file path

dataJSON = "journalsabbr.json"
journal_abbreviation_dict = {}
with open(dataJSON) as ff:
    journal_abbreviation_dict = json.load(ff)["full2Abbr"]

process_bibtex_file(input_bib_file, output_bib_file, journal_abbreviation_dict)





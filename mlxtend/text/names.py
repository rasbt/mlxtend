import unicodedata
import string
import re

def generalize_names(name, output_sep=' ', firstname_output_letters=1):
    """
    Function that outputs a person's name in the format 
    <last_name><separator><firstname letter(s)> (all lowercase)
        
    Parameters
    ----------
    name : `str`
      Name of the player
    output_sep : `str` (default: ' ')
      String for separating last name and first name in the output.
    firstname_output_letters : `int`
      Number of letters in the abbreviated first name.
      
    Returns
    ----------
    gen_name : `str`
      The generalized name.
        
    """
    # set first and last name positions
    last, first = 'last', 'first'
    last_pos = -1
    
    name = name.lower()
    
    for n in ('van der ', 'de ', 'van ', 'von ', 'ben '):
        if n in name:
            name = name.replace(n, n.replace(' ', ''))
            break
    
    if ',' in name:
        last, first = first, last   
        name = name.replace(',', ' ')
        last_pos = 1
        
    spl = name.split()
    if len(spl) > 2:
        name = '%s %s' % (spl[0], spl[last_pos])    

    # remove accents
    name = ''.join(x for x in unicodedata.normalize('NFKD', name) if x in string.ascii_letters+' ')
    
    # get first and last name if applicable
    m = re.match('(?P<first>\w+)\W+(?P<last>\w+)', name)
    if m:
        output = '%s%s%s' % (m.group(last), output_sep, m.group(first)[:firstname_output_letters])
    else:
        output = name
        
    gen_name = output.strip()
    return gen_name
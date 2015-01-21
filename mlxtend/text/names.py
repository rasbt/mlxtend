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
    
    # fix primarily Dutch names
    exc = ['van der ', 'de ', 'van ', 'von ', 'di ']
    for e in exc:
        if name.startswith(e):
            repl = e.replace(' ','')
            name = (repl + name[len(e)-1:].strip())
            
    exc = [' van der ', ' de ', ' van ', ' von ', ' di ',
    ', van der ', ', de', ', van ', ', von ', ', di ']
    
    for e in exc:
        name = name.replace(e, ' '+e.replace(' ', ''))
            
        
    if ',' in name:
        last, first = first, last   
        name = name.replace(',', '')
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
    
    
def generalize_names_duplcheck(df, col_name):
    """
    Applies mlxtend.text.generalize_names to a DataFrame with 1 first name letter
    by default and uses more first name letters if duplicates are detected.
        
    Parameters
    ----------
    df : `pandas.DataFrame`
      DataFrame that contains a column where generalize_names should be applied.
    
    col_name : `str` 
      Name of the DataFrame column where `generalize_names` function should be applied to.
    
    Returns
    ----------
    df_new : `str`
      New DataFrame object where generalize_names function has been applied without duplicates.
        
    """
    df_new = df.copy()
    
    df_new.drop_duplicates(subset=[col_name], inplace=True)
    
    df_new[col_name] = df_new[col_name].apply(generalize_names)
    
    dupl = list(df_new[df_new.duplicated(subset=col_name, take_last=True)].index) + \
       list(df_new[df_new.duplicated(subset=col_name, take_last=False)].index)
        
    firstname_letters = 2
    while len(dupl) > 0:
        for idx in dupl:
            df_new.loc[idx, col_name] = generalize_names(df.loc[idx, col_name], 
                                                  firstname_output_letters=firstname_letters)
        dupl = list(df_new[df_new.duplicated(subset=col_name, take_last=True)].index) + \
               list(df_new[df_new.duplicated(subset=col_name, take_last=False)].index)
        firstname_letters += 1
    return df_new
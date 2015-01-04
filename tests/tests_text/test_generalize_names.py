from mlxtend.text import generalize_names

def test_generalize_names():
    
    assert(generalize_names("Samuel Eto'o") == 'etoo s') 
    assert(generalize_names("Eto'o, Samuel") == 'etoo s') 
    assert(generalize_names("Eto'o, Samuel") == 'etoo s') 
    assert(generalize_names('Xavi') == 'xavi') 
    assert(generalize_names('Yaya Touré') == 'toure y') 
    assert(generalize_names('Pozo, José Ángel') ==  'pozo j') 
    assert(generalize_names('Pozo, José Ángel') == 'pozo j') 
    assert(generalize_names('José Ángel Pozo') == 'pozo j') 
    assert(generalize_names('José Pozo') == 'pozo j') 
    assert(generalize_names('Pozo, José Ángel', firstname_output_letters=2) == 'pozo jo') 
    assert(generalize_names("Eto'o, Samuel", firstname_output_letters=2) == 'etoo sa') 
    assert(generalize_names("Eto'o, Samuel", firstname_output_letters=0) == 'etoo') 
    assert(generalize_names("Eto'o, Samuel", output_sep=', ') == 'etoo, s') 
    assert(generalize_names("Eto'o, Samuel", output_sep=', ') == 'etoo, s') 

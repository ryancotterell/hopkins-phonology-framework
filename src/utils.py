from words import TemplaticWord
from words import TwoMorphemeWord
import fst
import numpy as np



def deep_copy_label_states(machine):
    new_machine = fst.LogVectorFst()
    for state_i,state in enumerate(machine):
        if state_i == 0:
            new_machine.start = new_machine.add_state()
        else:
            new_machine.add_state()

    
        if state.final != fst.LogWeight.ZERO:
            new_machine[state_i].final = fst.LogWeight.ONE

    for state_i,state in enumerate(machine):
        for arc_i,arc in enumerate(state):
            new_machine.add_arc(state_i,arc.nextstate,arc.ilabel,arc.olabel,fst.LogWeight(arc.nextstate))

    return new_machine

def deep_copy(machine):
    new_machine = fst.LogVectorFst()
    for state_i,state in enumerate(machine):
        if state_i == 0:
            new_machine.start = new_machine.add_state()
        else:
            new_machine.add_state()
        #new_machine[state_i].final = fst.LogWeight.ONE
        if state.final != fst.LogWeight.ZERO:
            new_machine[state_i].final = fst.LogWeight.ONE
    
    for state_i,state in enumerate(machine):
        for arc_i,arc in enumerate(state):
            new_machine.add_arc(state_i,arc.nextstate,arc.ilabel,arc.olabel,0.0)

    return new_machine

def peek(wfst,n):
    if isinstance(wfst,fst.LogVectorFst):
        wfst = fst.StdVectorFst(wfst,wfst.isyms,wfst.osyms)
    
    #wfst.remove_epsilon()
    #wfst = wfst.determinize()
    shortest_paths = wfst.shortest_path(n=n)
    
    shortest_paths.remove_epsilon()
    #shortest_paths = shortest_paths.determinize()

    shortest_paths_log = fst.LogVectorFst(shortest_paths,wfst.isyms,wfst.osyms)

    Z = np.exp(-float(shortest_paths_log.shortest_distance(True)[0]))
    #Z = 1.0 #fst.LogWeight.ONE
    strings = []

    wfst.osyms = wfst.isyms
    
    for path_i,path in enumerate(shortest_paths.paths()):
        
        string = ""

        weight = fst.TropicalWeight.ONE
        last_arc = None
        for arc in path:
            weight *= arc.weight
            if arc.ilabel > 0:
                string += wfst.osyms.find(arc.ilabel)

            last_arc = arc
        if len(path) > 0:
            weight *= shortest_paths[last_arc.nextstate].final
        else:
            weight *= shortest_paths[0].final
        strings.append((string,np.exp(-float(weight)) / Z ))
    strings.sort(key=lambda x: -x[1])

    for string,weight in strings:
        print weight,string


# Read in data
def read_data_arabic(data_in):
    data_in = open(data_in,'r')

    # SYMBOLS
    VOWELS = ["a","u","i","A","U","I"]
    #CONSONANTS = ["q","t","b","f","h","m","l","'"]
    CONSONANTS = []

    words = []
    # READ IN DATA (MOVE)
    for line_i,line in enumerate(data_in):
        line = line.rstrip("\n")
        #if line_i >= 24:
        #    break
        line = line.replace("_","")
        split = line.split("\t")
        sr = split[0]
        root = split[2]
        pattern = split[3]
        suffix = split[4]

        word = TemplaticWord(sr, root,pattern,suffix)
        words.append(word)

        for char in list(sr):
            if char not in VOWELS and char not in CONSONANTS:
                CONSONANTS.append(char)
            
    data_in.close()
    return words,VOWELS,CONSONANTS


# Read in data
def read_data_standard(data_in):
    data_in = open(data_in,'r')

    words = []
    # READ IN DATA (MOVE)
    for line_i,line in enumerate(data_in):
        line = line.rstrip("\n")
        split = line.split("\t")
        morpheme1 = split[0]
        morpheme2 = split[1]

        word = TwoMorphemeWord(morpheme1,morpheme2)
        words.append(word)

            
    data_in.close()
    return words



def make_templatic_alphabets(consonants, vowels):
    sigma = fst.SymbolTable()
    vowel_set = set([])
    consonant_set = set([])
    # add breaker
    sigma["#"] = 1 
    counter = 2

    for v in vowels:
        vowel_set.add(v)
        sigma[v] = counter
        counter +=1 
    for c in consonants:
        consonant_set.add(c)
        sigma[c] = counter
        counter += 1


    delta = fst.SymbolTable()
    delta["C"] = 1
    delta["V"] = 2

    return sigma,delta


def make_splitter():
    
    splitter = fst.LogTransducer(sigma,sigma)
    splitter.add_state()
    splitter[1].final = True

    for k,v in sigma.items():
        if v > 1:
            splitter.add_arc(0,0,k,k,0.0)

    splitter.add_arc(0,1,fst.EPSILON,"#",0.0)
    
    for k,v in sigma.items():
        if v > 1:
            splitter.add_arc(1,1,k,k,0.0)


    return splitter



def phonology_edit(sigma, COPY=.9):
    REMAINDER = (1.0 - COPY)
    edit = fst.LogTransducer(sigma, sigma)
    edit[0].final = True
    for k1, v1 in sigma.items():
        if v1 == 0:
            continue
        for k2, v2 in sigma.items():
            if v2 == 0:
                continue

            # substitutions
            if v1 == v2:
                edit.add_arc(0, 0, k1, k2, -np.log(COPY))
            else:
                edit.add_arc(0, 0, k1, k2, -np.log( (1.0-COPY) / (2*(len(sigma)-1))))
                
        # deletions
        edit.add_arc(0, 0, k1, fst.EPSILON, -np.log( (1.0-COPY) / (2*(len(sigma)-1))))
        
        # insertions
        edit.add_arc(0, 0, fst.EPSILON, k1, -np.log( (1.0-COPY) / (2*(len(sigma)-1))))

    return edit



def ngrams(machine,tolerance,max_len,pre_approved=set([])):
    alphas = machine.shortest_distance()
    betas = machine.shortest_distance(True)

    counts = {}
    cur_i = 0
    queue = []
    for state_i in range(1,len(machine)):
        queue.append((state_i,"",alphas[state_i]))
    queue.append((cur_i,"$",fst.LogWeight.ONE))


    while len(queue) > 0:
        cur_i,string,prob = queue.pop()
        for arc in machine[cur_i]:
            new_i = arc.nextstate
            new_prob = prob * arc.weight
            new_string = string
            

            if arc.ilabel > 0:
                new_string = string + machine.isyms.find(arc.ilabel)
            
            if (float(new_prob) < tolerance and len(new_string) < max_len and arc.ilabel > 0) or (new_string in pre_approved and arc.ilabel > 0):
                if cur_i != new_i:
                    queue.append((new_i,new_string,new_prob))
                    
                    
            # add to queue
            if float(new_prob) < tolerance or new_string in pre_approved:
                if new_string not in counts:
                    counts[new_string] = fst.LogWeight.ZERO
                
                counts[new_string] += (new_prob * betas[new_i])
            

            if machine[cur_i].final != fst.LogWeight.ZERO:
                final_prob = prob * arc.weight * betas[cur_i]
                final_string = string + "%"
                
                if float(final_prob) < tolerance or final_string in pre_approved:
                    if final_string not in counts:
                        counts[final_string] = fst.LogWeight.ZERO
                
                    counts[final_string] += final_prob

            
                

    return counts
        
    
        
    
def extract_pre_approved(machines):
    pre_approved = set([])
    for m in machines:
        # k-best
        for path in fst.StdVectorFst(m).shortest_path(n=1).paths():
            path_str = "".join(["" if x.ilabel == 0 else m.isyms.find(x.ilabel) for x in path]) 
            
            #pre_approved.add(path_str[-4:])
            pre_approved.add(path_str[-3:])
            pre_approved.add(path_str[-2:])
            


    return pre_approved

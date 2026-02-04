import numpy as np
import qlat as q

class Expression:
    def __init__(self, expr_string):
        self.expr = expr_string

        self.op_src = None #these are operator objects that are present in the expression. This expression object will parse vevs, two point functions, and three point functions. 
        self.op_snk = None
        self.op_ins = None
        self.wf_src = 0
        self.wf_snk = 0
        self.smear_src = 0
        self.smear_snk = 0
        self.idx = 0
        self.diagram_type = None
        self.num_ops = 0
        self.is_counter = False
        
        pass

    #method takes in the raw string for each expression, parses it, and returns a correlator object
    #corresponding to the number of operators involved. This method applies the wavefunction prefactors
    # and defines the operators present in the expression based on the input string, then builds a correlator from it. 
    def parse_expr(self):
        #the strings have three main components, an angle bracket section where the wave functions and operators are, a diagram type, and an expression index. 
        s = self.expr

        match = re.search(r'<(.*)>', s)
        if not match:
            return None

        #do not forget to add the counter special case here.
        
        angle = re.compile(r'<(.*)>')
        diagram_type = re.compile(r'\((ADT[\d_]+)\)')
        index = re.compile(r'exprs\[(\d+)\]')
        
        corr_str = angle.search(s).group(1) #the 1 here ensures that the angle brackets are not captured. This string is then just the text in the angle bracket

        if corr_str == '1':
            #define counter correlator
            self.handle_counter()
        
        diagram_str = diagram_type.search(s).group(1)
        index_str = index.search(s).group(1) 

        self.idx = int(index_str)
        self.diagram_type = diagram_str

        #we must further break up the corr_str, by asterisks. this returns a list of each individual item. 
        corr_term_list = corr_str.split(" * ")
        self.term_dict_list = []
        
        for term in corr_term_list:
            self.term_dict_list.append(self.parse_term(term))   
            print(self.parse_term(term))

        if self.num_ops == 0:
            #deal with this case later
            print("No operators in expression.")

        # when we parse a term, we count how many of them are operators. From here,
        # we will define operator objects, and then along with the operator information
        # and diagram and index information, we will construct a correlator object, and return it
        
        elif self.num_ops == 1:
            self.one_op_from_terms()

        elif self.num_ops == 2:
            self.two_ops_from_terms()
        
        elif self.num_ops == 3:
            self.three_ops_from_terms()

        
        
        #we need to deal with the fringe case of the counter expression, and the case of having 0 1 or 2 momentum projections

    #method that reads in a term from the correlator term list and returns a dictionary of the information it contains.
    #For wf terms we wish to save the fact that it is a wf term, its projection number, and if it is for the source or sink
    #for pipi terms we want to save the name and if it is a source or snk. 
    def parse_term(self,corr_term:str): 

        #capture the first bit of text in the string, then optionally capture the ^dag piece of the source operator term
        match = re.search(r'(\w+)(?:\^dag)?\(([^)]*)\)', corr_term)
        if not match:
            return {'type': 'unknown', 'raw': term}

        name = match.group(1) #this will be the name of the term no matter the type of term
        args = match.group(2) # this will give us the argument in the parenthesis, important for discerning the number of units of momentum. 

        #handle the case of no target specified (vevs are src and snk)
        if name.startswith('wf'):
            
            if len(name) == 2:
                role == None
            else:
                role = name[3:] #we want this to be src, snk, or int. 
            
            try:
                mom = int(args)
                return { 
                    'type': 'wavefunction',
                    'subtype': 'momentum',
                    'role': role,
                    'p': mom,
                    'raw': corr_term,
                }
            except ValueError:
                return {'type': 'unknown', 'raw': term}
            
        elif name.startswith('sm'):
            if len(name) == 2:
                role == None
            else:
                role = name[3:] #we want this to be src, snk, or int. 
            
            try:
                rad = int(args)
                return {
                    'type': 'wavefunction',
                    'subtype': 'smearing',
                    'role': role,
                    'r': rad,
                    'raw': corr_term,
                }
            except ValueError:
                return {'type': 'unknown', 'raw': term}
            
        #if the term is not a wavefunction term, it is an operator. 
        else: 
            is_dagger = ('^dag' in corr_term)
            is_subtracted = ('sub' in corr_term)

            #check if its a vev first, the single op acts as a source and sink. The argument should always be the source timeslice.
            if 'tsrc' in args:
                role = 'vev'
            
            #if its not a vev, the sink operator will always be the first operator present. We add a bit more to make sure
            elif self.num_ops == 0 or ('tsep' in args) or ('+' in args):
                role = 'snk'

            #the source operator will always have tsep = 0 due to translational invariance
            elif args == '0':
                role = 'src'

            #everything else we will treat as an intermediate operator.
            else:
                role = 'int'

            self.num_ops += 1 #counts operators so we know what to look for later
                
            return {
                'type': 'operator',
                'name': name,
                'role': role,
                'is_dagger': is_dagger,
                'is_subtracted': is_subtracted,
                'raw': corr_term,
            }






var_index={}
index = 1

num_variables = 0
num_clauses = 0

def if_and_only_if(ast, ac, bst, bc):
    global num_clauses

    iff_str = ""
    for v1 in range(5):
        temp_c = f"{ast+5*v1+ac} {-bst-5*v1-bc} 0\n"
        iff_str +=temp_c
        num_clauses += 1

        temp_c = f"{-ast-5*v1-ac} {bst+5*v1+bc} 0\n"
        iff_str +=temp_c
        num_clauses += 1

    return iff_str


def to_left(ast, ac, bst, bc):
    global num_clauses

    iff_str = ""
    
    temp_c = f"{-ast-4*5-ac} 0\n"
    iff_str += temp_c
    num_clauses+=1

    for v1 in range(4):
        temp_c = f"{ast+5*v1+ac} {-bst-5*(v1+1)-bc} 0\n"
        iff_str +=temp_c
        num_clauses += 1

        temp_c = f"{-ast-5*v1-ac} {bst+5*(v1+1)+bc} 0\n"
        iff_str +=temp_c
        num_clauses += 1

        return iff_str

def neighbor(ast,ac,bst,bc):
    global num_clauses

    iff_str = ""

    temp_c = f"{-ast-ac} {bst+5+bc} 0\n"
    iff_str +=temp_c
    num_clauses+=1

    temp_c = f"{-ast-4*5-ac} {bst+3*5+bc} 0\n"
    iff_str +=temp_c
    num_clauses+=1

    for v1 in range(1,4):
         
        temp_c = f"{-ast-5*v1-ac} {bst+5*(v1-1)+bc} {bst+5*(v1+1)+bc} 0\n"
        iff_str += temp_c
        num_clauses += 1

    return iff_str


def create_encoding():
    global num_clauses
    global num_variables

    with open(fname,"w") as f:
        # unique house color/national

        for constr in range(5):
            # at least one color/national to each house
            for i in range(5):
                temp_c = ""
                for c in range(5):
                    temp_c += str(i*5+1+c+25*constr) + " "
                    num_variables +=1

                temp_c += "0"
                f.write(temp_c+"\n")
                num_clauses +=1

            # exactly one color/national to each house
            for h in range(5):
                for i in range(5):
                    for c in range(i+1,5):

                        temp_c = str(-5*h-i-1-25*constr) + " " + str(-5*h-c-1-25*constr) + " 0"
                        f.write(temp_c+"\n")
                        num_clauses+=1

            
            # no to house have same color/national
            for h1 in range(5):
                for h2 in range(h1+1,5):
                    for j in range(5):

                        temp_c = str(-5*h1-j-1-25*constr) + " "+str(-5*h2-j-1-25*constr) + " 0"
                        f.write(temp_c+"\n")
                        num_clauses+=1

            
        #Hint 1
        f.write(if_and_only_if(var_index["color"],0,var_index["national"],0))

        #Hint 2
        f.write(if_and_only_if(var_index["national"],1,var_index["pet"],0))

        #Hint 3
        f.write(if_and_only_if(var_index["national"],2,var_index["drink"],0))

        #Hint 4
        f.write(to_left(var_index["color"],1,var_index["color"],2))

        #Hint 5
        f.write(if_and_only_if(var_index["color"],1,var_index["drink"],1))

        #Hint 6
        f.write(if_and_only_if(var_index["cigar"],0,var_index["pet"],1))

        #Hint 7
        f.write(if_and_only_if(var_index["color"],3,var_index["cigar"],1))

        #Hint 8
        f.write(str(var_index["drink"]+2*5+2)+" 0\n")
        num_clauses+=1

        #Hint 9
        f.write(str(var_index["national"]+3)+" 0\n")
        num_clauses+=1

        #Hint 10 neighbor
        f.write(neighbor(var_index["cigar"],2,var_index["pet"],2))

        #Hint 11 neighbor
        f.write(neighbor(var_index["pet"],3,var_index["cigar"],1))

        #Hint 12
        f.write(if_and_only_if(var_index["cigar"],3,var_index["drink"],3))

        #Hint 13
        f.write(if_and_only_if(var_index["national"],4,var_index["cigar"],4))

        #Hint 14 neighbor
        f.write(neighbor(var_index["national"],3,var_index["color"],4))

        #Hint 15 neighbor
        f.write(neighbor(var_index["cigar"],2,var_index["drink"],4))





if __name__ == "__main__":
    
    fname="einstein.txt"

    # clauses = []

    # house color: 1 - 25
    # house national: 26 - 50
    # house pet: 51 - 75
    # house drink: 76 - 100
    # house cigar: 101 - 125
    var_index={
        "color":1,
        "national":26,
        "pet": 51,
        "drink":76,
        "cigar":101
    }

    create_encoding()
    print(num_clauses)
    print(num_variables)

    #prepend above
    with open(fname, 'r') as original: data = original.read()
    with open(fname, 'w') as modified: modified.write(f"p cnf {num_variables} {num_clauses}\n" + data)
    


    

        










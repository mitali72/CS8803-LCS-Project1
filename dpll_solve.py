import numpy as np
from copy import deepcopy
import argparse

np.random.seed(42)

class Assignment:
  def __init__(self, N):
    # assignment of values to variables
    # assignments[i] == 1 => x_{i+1} is True
    # -1 for False
    # 0 for unassigned
    self.assignment = np.zeros(N)
    self.N = N

  def assign(self, i, val):
    # ith var (in 1-indexing in line with cnf nomenclature) is given value val
    if i==0:
      raise Exception
    self.assignment[i-1] = val

  def key_palette(self):
    # converts assignment to a replacement dict
    # so assignment = [1, 0, -1, 0] gives key = [0, 1, 0, -1, -1, 1, 0, -1, 0]
    #   note that variable '0' is assigned -1 (False)
    # for a palette [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    palette = np.array([i+1 for i in range(self.N)]) # 1, 2, ... N
    # -N, -N+1, .., 1, 1, 2, ... N
    palette = np.concatenate((-palette[::-1], [0], palette))
    key = np.concatenate((-self.assignment[::-1], [-1], self.assignment))
    return key, palette

class Evaluation:
  # evaluation of a cnf under an assignment
  def __init__(self, cnf, assignment):
    self.cnf = cnf
    self.assignment = assignment
    # each variable is evaluated as per assignment
    # if assignment is undefined (0) or if the variable is undefined(0)
    #   then the evaluation is -1
    key, palette = assignment.key_palette()
    # mapping cnf matrix to assignment values
    index = np.digitize(cnf.cnf_matrix.ravel(), palette, right=True)
    self.evaluation = key[index].reshape(cnf.cnf_matrix.shape)

  def _is_complete(self, verbose=False):
    # if any clause is competely -1 => cnf evaluates to False, returns False
    # if all clauses have atleast one +1 => cnf evaluates to True, returns True
    # else, returns None, and simplified evaluation is saved
    
    minus_one_counts = (self.evaluation==-1).sum(axis=1)
    if sum(minus_one_counts==self.cnf.K)>0:
      if(verbose):
        print("Evaluated as false at the clause:")
        print(self.cnf.cnf_matrix[(minus_one_counts==self.cnf.K), :])
      return False

    plus_one_counts = (self.evaluation==1).sum(axis=1)
    if sum(plus_one_counts>0)==self.cnf.L:
      if verbose:
        print("Evaluated as true")
      return True

    return None

  def simplify(self, verbose=False):
    # returns simplified cnf, and if the simplified cnf is just True/False
    #   then returns None. Whether the simplified cnf was true/false can be checked
    #   through self.is_complete
    self.is_complete = self._is_complete(verbose)


    if self.is_complete is None:
      # filter out clauses that are True
      clauses_left = (self.evaluation==1).sum(axis=1)==0
      new_cnf_matrix = self.cnf.cnf_matrix.copy()
      new_cnf_matrix[self.evaluation!=0] = 0
      new_cnf_matrix = new_cnf_matrix[clauses_left, :]
      new_cnf = CNF(self.cnf.N, clauses_left.sum(), self.cnf.K)
      new_cnf.cnf_matrix = new_cnf_matrix
      return new_cnf
    return None


class CNF:
  def __init__(self, N, L, K=3):
    self.N = N # number of variables
    self.L = L # number of clauses
    self.K = K # max number of variables per clause
  
    # cnf is represneted as matrix
    # i th row is i th clause
    # j th number is index of jth variable
    # 8 = x_8, -8 = \bar{x_8}
    # 0 = no variable
    self.cnf_matrix = np.zeros((L, K))

  def initialise_matrix(self, clause_list):
    # input is a list of list of variables
    for i, clause in enumerate(clause_list):
      for j, var in enumerate(clause):
        self.cnf_matrix[i, j] = var

  def initialise_matrix_randomly(self, seed=0):
    # initialize cnf randomly, as described in the problem statement
    def sample_noreplace(arr, l, k):
      # randomly choose k elements out of arr, l times
      idx = np.random.randint(len(arr) - np.arange(k), size=[l, k])
      for i in range(k-1, 0, -1):
          idx[:,i:] += idx[:,i:] >= idx[:,i-1,None]
      return np.array(arr)[idx]

    np.random.seed(seed)
    variables_matrix = sample_noreplace(np.arange(1, self.N+1), self.L, self.K)
    sign_matrix = np.random.choice([-1, 1], size=(self.L, self.K))
    self.cnf_matrix = variables_matrix*sign_matrix

  def to_string(self):
    s = ""
    s += f"p cnf {self.N} {self.L}\n"
    for i in range(self.cnf_matrix.shape[0]):
      clause  = self.cnf_matrix[i, :]
      s += f"{' '.join(map(str, np.int8(clause[clause!=0])))} 0\n"
    return s  

    
def read_file_and_make_cnf(filename):
  with open(filename) as f:
    lines = f.readlines()

  # remove comments, and pick out cnf format line
  filtered_lines = []
  format_line = None
  for line in lines:
    if line.startswith("c"):
      continue
    if line.startswith("p"):
      format_line = line
      continue
    filtered_lines.append(line)

  if format_line is None:
    raise Exception("No format line found that begins with 'p'")

  if len(filtered_lines)==0:
    raise Exception("No non-comment lines found")
  
  # get N, L
  N = int(format_line.split()[2])
  L = int(format_line.split()[3])

  # process clauses
  all_clauses_as_one_str = " ".join(map(str.strip, filtered_lines))
  clauses = list(map(lambda x: x.strip().split(),
    all_clauses_as_one_str.split(" 0")))

  non_empty_clauses = []
  for clause in clauses:
    if len(clause)>0:
      non_empty_clauses.append(clause)

  K = max(map(len, non_empty_clauses))
  cnf = CNF(N, L, K)
  cnf.initialise_matrix(non_empty_clauses)
  return cnf

class Heuristic:
  # class that any implemented heuristic should inherit and implement the methods of
  def __init__(self):
    # list of tuples (variable, value, toggled), used as a stack
    self.assignment_history = []

    # # indicating that the first assignment has not been made
    # self.first_assignment_pending = True
  
  def unit_preference_rule(self, unit_clauses):
    # unit_clauses is a 1d array of unit clauses
    # so for the CNF (x_2).(x_3+x_4).(\bar{x_5}): unit_clauses = [2, -5]
    # return an assignment tuple to set the truth value of
    #   a proposition from the set of unit clauses (eg. (5, -1))

    var = np.abs(unit_clauses[0])
    val = np.sign(unit_clauses[0])
    return int(var), int(val)

  def get_unit_clauses(self, cnf_matrix):
    # returns a 1d array of unit clauses
    # unit clause if K-1 variables are 0
    is_unit_clause = ((cnf_matrix==0).sum(axis=1) == (self.cnf.K - 1))
    return cnf_matrix[is_unit_clause, :].sum(axis=1)

  def take_one_step(self):
    # if unit clauses exist, call unit_preference_rule
    # else call splitting_rule
    # and return an assignment tuple
    unit_clauses = self.get_unit_clauses(self.new_cnf.cnf_matrix)
    if len(unit_clauses) > 0:
      return self.unit_preference_rule(unit_clauses)
    return self.splitting_rule(self.new_cnf.cnf_matrix)

  def solve_dpll(self, cnf, verbose=False):
    self.cnf = cnf
    self.assignment = Assignment(self.cnf.N)
    self.verbose = verbose

    self.new_cnf = deepcopy(cnf)
    return self.dpll_recursive()


  def dpll_recursive(self):
    var, val = self.take_one_step()

    # first branch
    if self.verbose:
      print(var, val, 1)
    self.assignment.assign(var, val)
    evaluation = Evaluation(self.cnf, self.assignment)
    self.new_cnf = evaluation.simplify(self.verbose)

    # is evaluation gives true, return the assignment (SAT)
    # if evaluation gives false, don't recurse further
    # else, there is scope for further recursion
    if self.new_cnf is None:
      if evaluation.is_complete:
        return self.assignment
      result1 = None
    else:
      result1 = self.dpll_recursive()
    
    # if SAT, don't look anymore
    if not (result1 is None):
      return result1

    # repeat for second branch
    if self.verbose:
      print(var, val, 2)
    self.assignment.assign(var, -val)
    evaluation = Evaluation(self.cnf, self.assignment)
    self.new_cnf = evaluation.simplify(self.verbose)

    if self.new_cnf is None:
      if evaluation.is_complete:
        return self.assignment
      result2 = None
    else:
      result2 = self.dpll_recursive()

    if not (result2 is None):
      return result2

    # reset assignment and go back a level, if there is any
    self.assignment.assign(var, 0)
    return None


class MyHeuristic(Heuristic):
  
  def splitting_rule(self, cnf_matrix):
    # does splitting and returns an assignemnt tuple
    # get the literal with max occurences in all clauses

    flattened_matrix = cnf_matrix.ravel()
    all_literals = np.unique(flattened_matrix)
    all_literals = all_literals[all_literals!=0]

    distinct_literals,counts = np.unique(all_literals, return_counts = True)

    max_indices, = np.where(counts==np.max(counts))
    if(len(max_indices)==1):
        return int(abs(distinct_literals[max_indices[0]])), int(np.sign(distinct_literals[max_indices[0]]))
    else:
        var = int(np.random.choice(max_indices,1)[0])
        return int(abs(distinct_literals[var])), int(np.sign(distinct_literals[var]))

class RandomHeuristic(Heuristic):

  def splitting_rule(self, cnf_matrix):
    # does splitting and returns an assignemnt tuple

    flattened_matrix = cnf_matrix.ravel()
    all_literals = np.unique(flattened_matrix)
    all_literals = all_literals[all_literals!=0]

    var = int(np.random.choice(all_literals,1)[0])
    return int(abs(var)), int(np.sign(var))
  

class TwoClauseHeuristic(Heuristic):

  def splitting_rule(self, cnf_matrix):
    # does splitting and returns an assignemnt tuple
    is_two_clause = ((cnf_matrix==0).sum(axis=1) == (self.cnf.K - 2))
    two_clause_props = cnf_matrix[is_two_clause,:]

    if len(two_clause_props)==0:
      #no clauses with just two literals, then do random
        flattened_matrix = cnf_matrix.ravel()
        all_literals = np.unique(flattened_matrix)
        all_literals = all_literals[all_literals!=0]

        var = int(np.random.choice(all_literals,1)[0])
        return int(abs(var)), 1
    
    else:
        two_clause_props = np.abs(two_clause_props)
        two_clause_props = two_clause_props[two_clause_props!=0]
        distinct_props,counts = np.unique(two_clause_props, return_counts = True)
        
        max_indices, = np.where(counts==np.max(counts))
        if(len(max_indices)==1):
            return int(distinct_props[max_indices[0]]),1
        else:
            var = int(np.random.choice(max_indices,1)[0])
            return int(distinct_props[var]),1
   

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Simulation of a P2P Cryptocurrency Network')
    parser.add_argument('--heuristic', help='select heuristic', default="random")

    args = parser.parse_args()

    #get cnf
    cnf = read_file_and_make_cnf("einstein.txt")
    # cnf = read_file_and_make_cnf("trials/trial_4_2.txt")
    if(args.heuristic=="random"):
      th = RandomHeuristic()

    elif(args.heuristic=="two_clause"):
      th = TwoClauseHeuristic()


    elif(args.heuristic=="my_heuristic"):
      th = MyHeuristic()
    
    else:
      print("invalid heuristic argument")
      exit
      
    #solve
    result = th.solve_dpll(cnf, False)
    if result is None:
        print("UNSAT")
    else:
        print("SATISFIABLE")
        # who owns fish
        inds = np.arange(1,126)
        print(result.assignment*inds)

#! /usr/bin/env python3
import os
import sys
import toml
import numpy as np
from pysat.solvers import Glucose3
import matplotlib.pyplot as plt
from matplotlib import cm

AUXVARS_LOWER_LIMIT = 10000
AUXVARS_INDEX = AUXVARS_LOWER_LIMIT

def at_least_one(inputs):
    return [inputs]

def at_most_one(inputs):
    clauses = []
    N = len(inputs)
    for i in range(0, N - 1):
        for j in range(i + 1, N):
            clauses.append([-inputs[i],-inputs[j]])
    return clauses

def at_most_one_sinz(inputs):
    global AUXVARS_INDEX

    N = len(inputs)

    clauses = []
    clauses.append([-inputs[0], AUXVARS_INDEX])
    clauses.append([-inputs[N-1], -(AUXVARS_INDEX+N-2)])

    for i in range(1,N-1):
        clauses.append([-inputs[i], AUXVARS_INDEX+i])
        clauses.append([-(AUXVARS_INDEX+i-1), AUXVARS_INDEX+i])
        clauses.append([-inputs[i], -(AUXVARS_INDEX+i-1)])
    AUXVARS_INDEX += 2*N

    return clauses

def one_hot_detector(inputs):
    clauses = []
    clauses.extend(at_least_one(inputs))
    clauses.extend(at_most_one_sinz(inputs))
    return clauses

def load_aztec_file(filename):
    with open(filename, "r") as f:
        puzzle_dict = toml.load(f)
    return puzzle_dict

def piece_rot90(piece):
    new_piece = []
    for pt in piece:
        new_piece.append([pt[1], -pt[0]])
    return new_piece

def piece_translate(piece, xy):
    new_piece = []
    for pt in piece:
        new_piece.append([pt[0]+xy[0], pt[1]+xy[1]])
    return new_piece

def piece_fits(puzzle, piece):
    for pt in piece:
        if pt[0] < 0 or pt[0] >= puzzle.shape[1]:
            return False
        if pt[1] < 0 or pt[1] >= puzzle.shape[0]:
            return False
        # Flip the y-axis (row) to avoid
        # solving the puzzle upside-down.
        if puzzle[puzzle.shape[0]-pt[1]-1,pt[0]] == 1:
            return False
    return True

def piece_covers(piece, xy):
    for pt in piece:
        if pt[0] == xy[0] and pt[1] == xy[1]:
            return True
    return False


def solve_aztec_puzzle(puzzle, pieces):

    print("Building placement table.")

    # Build the placement table.
    placements = [None]
    for piece_idx, piece in enumerate(pieces):
        rotated_piece = piece
        for rotation_idx in range(3):
            for x in range(puzzle.shape[1]):
                for y in range(puzzle.shape[0]):
                    placed_piece = piece_translate(rotated_piece, [x,y])
                    if piece_fits(puzzle, placed_piece):
                        placements.append((piece_idx, placed_piece))
            rotated_piece = piece_rot90(rotated_piece)

    print("Constructing puzzle square constraints.")

    # Construct one-hot constraints requiring only one piece per puzzle location.
    one_hots = []
    for x in range(puzzle.shape[1]):
        for y in range(puzzle.shape[0]):
            print(x,y)
            one_hot = []
            for place_idx, placement in enumerate(placements[1:]):
                piece = placement[1]
                if piece_covers(piece, [x,y]):
                    one_hot.append(place_idx+1)
            if len(one_hot) > 0:
                one_hots.append(one_hot)

    print("Constructing puzzle piece constraints.")

    # Construct one-hot constraints requiring each piece only be used once.
    piece_hash = {}
    for place_idx, place in enumerate(placements[1:]):
        piece_idx = place[0]
        if(piece_idx not in piece_hash):
            piece_hash[piece_idx] = []
        piece_hash[piece_idx].append(place_idx+1)

    for piece_idx, one_hot in piece_hash.items():
        one_hots.append(one_hot)

    print("Constructing the solver.")

    solver = Glucose3()
    seent = set()
    num_terms = 0
    duplicates = 0
    for i,oh in enumerate(one_hots):
        print("{}/{} {:0.3f}".format(i, len(one_hots), 100*i/len(one_hots)))
        ohd = one_hot_detector(oh)
        print(len(ohd))
        for term in ohd:
            num_terms += 1
            term = tuple(term)
            if term not in seent:
                solver.add_clause(term)
            else:
                duplicates += 1
            seent.add(term)
        print("BEFORE: {} AFTER: {} REDUCTION: {:0.3f}".format(num_terms, num_terms-duplicates, num_terms/(1+num_terms-duplicates)))

    print("Solving!")

    # Solve!
    solutions = []
    if solver.solve():
        print("Solution Found!")
        for model in solver.enum_models():
            model = np.array(model)
            model = model[model > 0]
            model = model[model < len(placements)]
            solution_placements = [placements[i] for i in model]
            solutions.append(solution_placements)
            break
    else:
        print("Solution Not Found.")
        return []

    return solutions

def plot_solution(puzzle, solution):
    puzzle = np.invert(puzzle)
    for piece_idx, piece in enumerate(solution):
        for pt in piece[1]:
            puzzle[puzzle.shape[0]-pt[1]-1, pt[0]] = piece_idx
    plt.imshow(puzzle, cmap=cm.inferno)
    plt.show()

if __name__ == "__main__":
    # Parse command line args.
    if len(sys.argv) < 2:
        print("Please provide a *.az aztec puzzle file.")
        print("usage: ./aztec.py puzzle.az")
        exit(0)
    filename = sys.argv[1]

    # Read the puzzle from a *.az file.
    puzzle_dict = load_aztec_file(filename)

    # Extract the puzzle and pieces from the puzzle dict.
    puzzle = np.array(puzzle_dict["puzzle"])
    pieces = [np.array(p) for p in puzzle_dict["pieces"]]

    print("Puzzle Dimensions: {}x{}".format(puzzle.shape[0], puzzle.shape[1]))
    print("Number of Pieces: {}".format(len(pieces)))

    # Check for a malformed puzzle.
    puzzle_area = puzzle.size - np.count_nonzero(puzzle)
    pieces_area = sum([p.shape[0] for p in pieces])

    if puzzle_area != pieces_area:
        print(
            "Error: Summed area of pieces ({}) does not match the puzzle free space ({}).".format(
                pieces_area, puzzle_area
            )
        )
        exit(0)

    # Solve the puzzle.
    print("Solving {}...".format(os.path.basename(filename)))
    solutions = solve_aztec_puzzle(puzzle, pieces)
    print("MADE IT HERE")
    if len(solutions) > 0:
        print("FOUND {} SOLUTIONS!".format(len(solutions)))
        for solution in solutions:
            plot_solution(puzzle, solution)
    else:
        print("FAILED TO SOLVE.")

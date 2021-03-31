#! /usr/bin/env python3
import os
import sys
import toml
import numpy as np
from pysat.solvers import Glucose3
import matplotlib.pyplot as plt
from matplotlib import cm


# The one_hot_detector function returns the CNF clauses describing a
# one-hot detector with the given input variables.
#
# For example, one_hot_detector([1,2,3]) will return
# [[-1,-2], [-1,-3], [-2, -3], [1,2,3]].
#
# Symbolically, this is equivalent to one_hot_detector([A, B, C]) returning
#    (~A | ~B) & (~A | ~C) & (~B | ~C) & ( A | B | C )
#
# I find the DNF form of the one-hot detector to be much more intuitive.
# This is the DNF equivalent.
#    (~A&~B&~C&D) | (~A&~B&C&~D) | (~A&B&~C&~D) | (A&~B&~C&~D)
def one_hot_detector(inputs):
    clauses = []

    N = len(inputs)
    for i in range(0, N - 1):
        for j in range(i + 1, N):
            clauses.append([-1 * inputs[i], -1 * inputs[j]])
    clauses.append(inputs)
    return clauses


# This function takes in a list of one_hot_detector outputs,
# concatenates their clauses, and returns the result.
# This is equivalent to ANDing multiple one_hot_detectors.
def one_hot_combiner(one_hots):
    combination = []
    for oh in one_hots:
        combination.extend(oh)
    return combination


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

    # Build the placement table.
    placements = []
    for piece_idx, piece in enumerate(pieces):
        rotated_piece = piece
        for rotation_idx in range(4):
            for x in range(puzzle.shape[1]):
                for y in range(puzzle.shape[0]):
                    placed_piece = piece_translate(rotated_piece, [x,y])
                    if piece_fits(puzzle, placed_piece):
                        placements.append((piece_idx, placed_piece))
            rotated_piece = piece_rot90(rotated_piece)

    # Construct one-hot constraints requiring only one piece per puzzle location.
    one_hots = []
    for x in range(puzzle.shape[1]):
        for y in range(puzzle.shape[0]):
            one_hot = []
            for place_idx, placement in enumerate(placements):
                piece = placement[1]
                if piece_covers(piece, [x,y]):
                    one_hot.append(place_idx)
            if len(one_hot) > 0:
                one_hots.append(one_hot)

    # Construct one-hot constraints requiring each piece only be used once.
    piece_hash = {}
    for place_idx, place in enumerate(placements):
        piece_idx = place[0]
        if(piece_idx not in piece_hash):
            piece_hash[piece_idx] = []
        piece_hash[piece_idx].append(place_idx)

    for piece_idx, one_hot in piece_hash.items():
        one_hots.append(one_hot)

    # Terms must be greater than zero, so bias them up by one.
    for oh in range(len(one_hots)):
        one_hots[oh] = [term+1 for term in one_hots[oh]]

    # Solve!
    solver = Glucose3()
    for clauses in one_hot_combiner([one_hot_detector(oh) for oh in one_hots]):
        solver.add_clause(clauses)

    solutions = []
    if solver.solve():
        print("Solution Found!")
        for model in solver.enum_models():
            model = np.array(model)
            model = model[model > 0]
            # Bias back down by one to use the terms 
            # as an index into the placement table.
            model -= 1
            solution_placements = [placements[i] for i in model]
            solutions.append(solution_placements)
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

def plot_piece(piece):
    for pt in piece:
        plt.scatter(pt[0], pt[1])
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

    # I use this to debug puzzles.
    #for p in pieces:
    #    plot_piece(p)

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
    if len(solutions) > 0:
        print("FOUND {} SOLUTIONS!".format(len(solutions)))
        for solution in solutions:
            plot_solution(puzzle, solution)
    else:
        print("FAILED TO SOLVE.")

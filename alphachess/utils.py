"""ChessWarrior utilities"""
import json
import logging
import os
import random
from functools import reduce

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)



def get_all_possible_moves():
    """return a list of all possible move in the chess"""
    """
    Creates the labels for the universal chess interface into an array and returns them
    """
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']

    for l1 in range(8):
        for n1 in range(8):
            destinations = [(t, n1) for t in range(8)] + \
                           [(l1, t) for t in range(8)] + \
                           [(l1 + t, n1 + t) for t in range(-7, 8)] + \
                           [(l1 + t, n1 - t) for t in range(-7, 8)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                    labels_array.append(move)
    for l1 in range(8):
        l = letters[l1]
        for p in promoted_to:
            labels_array.append(l + '2' + l + '1' + p)
            labels_array.append(l + '7' + l + '8' + p)
            if l1 > 0:
                l_l = letters[l1 - 1]
                labels_array.append(l + '2' + l_l + '1' + p)
                labels_array.append(l + '7' + l_l + '8' + p)
            if l1 < 7:
                l_r = letters[l1 + 1]
                labels_array.append(l + '2' + l_r + '1' + p)
                labels_array.append(l + '7' + l_r + '8' + p)
    return labels_array

PIECES_ORDER = 'KQRBNPkqrbnp'
PIECES_INDEX = {PIECES_ORDER[i] : i for i in range(12)}
EXTEND_SPACE = {
'1' : '1',
'2' : '11',
'3' : '111',
'4' : '1111',
'5' : '11111',
'6' : '111111',
'7' : '1111111',
'8' : '11111111',
'/' : ''
}

labels = get_all_possible_moves()
label_len = len(labels)

def is_black_turn(fen):
    return fen.split(' ')[1] == 'b'

def get_board_string(board_fen_0):

    rows = board_fen_0.split('/')
    board_string = reduce(lambda x, y: x + y,
                          list(reduce(lambda x, y: x + y, list(map(lambda x: x if x.isalpha() else EXTEND_SPACE[x], row)))
                        for row in rows))
    assert len(board_string) == 64
    return board_string

def get_history_plane(board_fen):

    board_fen_list = board_fen.split(' ')

    history_plane = torch.zeros(12, 8 , 8)

    board_string = get_board_string(board_fen_list[0])

    for i in range(8):
        for j in range(8):
            piece = board_string[(i << 3) | j]
            if piece.isalpha():
                history_plane[PIECES_INDEX[piece]][i][j] = 1
    return history_plane


def fen_positon_to_my_position(fen_position):
    return 8 - int(fen_position[1]), ord(fen_position[0]) - ord('a')

def get_auxilary_plane(board_fen):

    board_fen_list = board_fen.split(' ')

    en_passant_state = board_fen_list[3]
    en_passant_plane = torch.zeros(8, 8)
    if en_passant_state != '-':
        position = fen_positon_to_my_position(en_passant_state)
        en_passant_plane[position[0]][position[1]] = 1
    fifty_move_count = eval(board_fen_list[4])
    fifty_move_plane = torch.full((8, 8), fifty_move_count)

    castling_state = board_fen_list[2]

    K_castling_plane = torch.full((8, 8), int('K' in castling_state))
    Q_castling_plane = torch.full((8, 8), int('Q' in castling_state))
    k_castling_plane = torch.full((8, 8), int('k' in castling_state))
    q_castling_plane = torch.full((8, 8), int('q' in castling_state))

    auxilary_plane = torch.stack((K_castling_plane, Q_castling_plane, k_castling_plane,
                               q_castling_plane, fifty_move_plane, en_passant_plane), dim=0)

    assert auxilary_plane.shape == (6, 8, 8)
    return auxilary_plane


def get_feature_plane(board_fen):

    history_plane = get_history_plane(board_fen)
    auxilary_plane = get_auxilary_plane(board_fen)
    feature_plane = torch.cat((history_plane, auxilary_plane), 0)
    assert feature_plane.shape == (18, 8, 8)
    return feature_plane

def first_person_view_fen(board_fen, flip):

    if not flip:
        return board_fen

    board_fen_list = board_fen.split(' ')
    rows = board_fen_list[0].split('/')

    rows = [reduce(lambda x, y : x + y, list(map(lambda ch: ch.lower() if ch.isupper() else ch.upper(), row))) for row in rows]
    board_fen_list[0] = '/'.join(reversed(rows))

    board_fen_list[1] = 'w' if board_fen_list[1] == 'b' else 'b'

    board_fen_list[2] = "".join(sorted("".join(ch.lower() if ch.isupper() else ch.upper() for ch in board_fen_list[2])))

    ret_board_fen = ' '.join(board_fen_list)
    return ret_board_fen

def first_person_view_move(move, flip):
    if not flip:
        return move
    
    new_move = []
    for s in move:
        if s.isdigit():
            new_move.append(str(9 - int(s)))
        else:
            new_move.append(s)
    return ''.join(new_move)

def convert_board_to_plane(board_fen):
    return get_feature_plane(first_person_view_fen(board_fen, is_black_turn(board_fen)))
    
class ChessDataset(Dataset):
    def __init__(self, config):
        self.dataset = []
        self.data_files = os.listdir(config.resources.sl_processed_data_dir)
        
        for data_file in self.data_files:
            with open(config.resources.sl_processed_data_dir + '/' + data_file, "r") as f:
                data = json.load(f)
            self.dataset.extend(data)
    
        all_moves = get_all_possible_moves()
        self.move_size = int(len(all_moves))
        self.move_hash = {move: i for (i, move) in enumerate(all_moves)}
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]

        if is_black_turn(data['s']):
            data['s'] = first_person_view_fen(data['s'], True)
            data['a'] = first_person_view_move(data['a'], True)

        data['s'] = get_feature_plane(data['s'])

        k = self.move_hash[data['a']]
        data['a'] = k
        
        data['r'] = torch.FloatTensor([data['r']])
        return data 

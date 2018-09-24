"""ChessWarrior utilities"""
import json
import logging
import os
import random
from functools import reduce
from tqdm import tqdm
from torchvision import transforms

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

def evaluate_board(fen):
    chess_piece_value = {'Q' : 14, 'R' : 5, 'B' : 3.25, 'K' : 3, 'N' : 3, 'P' : 1}
    current_value = 0.0
    total_value = 0.0
    for ch in fen.split(' ')[0]:
        if not ch.isalpha():
            continue
        if ch.isupper():
            current_value += chess_piece_value[ch]
            total_value += chess_piece_value[ch]
        else:
            current_value -= chess_piece_value[ch.upper()]
            total_value += chess_piece_value[ch.upper()]

    value_rate = current_value / total_value

    return float(np.tanh(value_rate))

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

    history_plane = np.zeros(shape=(12, 8, 8), dtype=np.float32)

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
    en_passant_plane = np.zeros((8, 8), dtype=np.float32)
    if en_passant_state != '-':
        position = fen_positon_to_my_position(en_passant_state)
        en_passant_plane[position[0]][position[1]] = 1
    fifty_move_count = eval(board_fen_list[4])
    fifty_move_plane = np.full((8, 8), fifty_move_count)

    total_move_count = eval(board_fen_list[5])
    total_move_plane = np.full((8, 8), total_move_count)

    castling_state = board_fen_list[2]

    K_castling_plane = np.full((8, 8), int('K' in castling_state), dtype=np.float32)
    Q_castling_plane = np.full((8, 8), int('Q' in castling_state), dtype=np.float32)
    k_castling_plane = np.full((8, 8), int('k' in castling_state), dtype=np.float32)
    q_castling_plane = np.full((8, 8), int('q' in castling_state), dtype=np.float32)
    
    board = chess.Board(board_fen)

    is_gameover_plnae = np.full((8, 8), int(board.is_game_over()), dtype=np.float32)
    is_checkmate_plane = np.full((8, 8), int(board.is_checkmate()), dtype=np.float32)
    is_stalemate_plane = np.full((8, 8), int(board.is_stalemate()), dtype=np.float32)
    is_insufficient_material_plane = np.full((8, 8), int(board.is_insufficient_material()), dtype=np.float32)
    is_seventyfive_moves_material_plane = np.full((8, 8), int(board.is_seventyfive_moves()), dtype=np.float32)
    is_fivefold_repetition_material_plane = np.full((8, 8), int(board.is_fivefold_repetition()), dtype=np.float32)

    mobility_plane = np.full((8, 8), len(board.legal_moves), dtype=np.float32)
    is_check_plane = np.full((8, 8), int(board.is_check()), dtype=np.float32)


    auxilary_plane = np.array([K_castling_plane, Q_castling_plane, k_castling_plane,
                               q_castling_plane, fifty_move_plane, en_passant_plane,
                               total_move_plane, is_gameover_plnae, is_checkmate_plane,
                               is_stalemate_plane, is_insufficient_material_plane, is_seventyfive_moves_material_plane,
                               is_fivefold_repetition_material_plane, mobility_plane, is_check_plane])

    assert auxilary_plane.shape == (15, 8, 8)
    return auxilary_plane


def get_feature_plane(board_fen):

    history_plane = get_history_plane(board_fen)
    auxilary_plane = get_auxilary_plane(board_fen)

    feature_plane = np.vstack((history_plane, auxilary_plane))
    assert feature_plane.shape == (27, 8, 8)
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
        dataset = []
        data_files = os.listdir(config.resources.sl_processed_data_dir)
        
        for data_file in data_files:
            with open(config.resources.sl_processed_data_dir + '/' + data_file, "r") as f:
                data = json.load(f)
            dataset.extend(data)
 
        all_moves = get_all_possible_moves()
        self.move_hash = {move: i for (i, move) in enumerate(all_moves)}
        
        self.dataset = dataset
        
        self.transform = transforms.Compose([GetFeatures()])
        
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        
        if is_black_turn(data['s']):
            data['s'] = first_person_view_fen(data['s'], True)
            data['a'] = first_person_view_move(data['a'], True)

        s = self.transform(data['s'])   
        a = self.move_hash[data['a']]
        r = np.array([data['r']], dtype=np.float32)        
        return s, a, r

class GetFeatures(object):
    def __call__(self, s):
        return get_feature_plane(s).astype(np.float32)

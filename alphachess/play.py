"""ChessWarrior play chess"""

import logging
import os
import matplotlib.pyplot as plt
import time
import multiprocessing

import numpy as np
import chess 
import torch

from .model import AlphaChess
from .config import Config
from .utils import convert_board_to_plane, get_all_possible_moves, first_person_view_fen, get_feature_plane, \
    is_black_turn


logger = logging.getLogger(__name__)

class Player(object):
    """Using the best model to play"""
    def __init__(self, config: Config):
        self.config = config
        self.model_path = self.config.resources.best_model_dir
        self.model = None
        self.value_model = None
        self.board = None
        self.choise = None
        self.search_depth = 4
        self.move_value = {}
        self.move_hash = {}
        self.policies = []
        self.cnt = []
        self.moves_cnt = 0

        self.used_time = 0
        self.INF = 0x3f3f3f3f

    def start(self):
        try:
            self.model = AlphaChess(self.config)
            weights = torch.load(os.path.join(self.config.resources.best_model_dir, "best_model.pth"))
            self.model.load_state_dict(weights['state_dict'])
            logger.info("Load best model successfully.")
        except OSError:
            logger.fatal("Model cannot find!")
            return

        self.board = chess.Board()
        
        all_moves = get_all_possible_moves()
        self.move_hash = {move: i for (i, move) in enumerate(all_moves)}
        
        while not self.board.is_game_over():
            move = self.play()
            self.board.push(move)
            print("AI move: ", move.uci())
            print(self.board)

            while True:
                try:
                    your_move = input()
                    self.board.push_uci(your_move)
                    break
                except Exception as e:
                    print(e)
                    continue
            print("Your move: ", your_move)
            print(self.board)
    
    def play(self):
        """
        return my move
        注意，到了这个函数，只需考虑自己是白方的
        相应转换工作在start函数已经完成，这里无需重复考虑
        """
        feature_plane = get_feature_plane(self.board.fen())
        feature_plane = feature_plane[np.newaxis, :].astype(np.float32)
        policy, v = self.model(torch.from_numpy(feature_plane))

        candidates = {}
        
        legal_moves = self.board.legal_moves
        for move in legal_moves:
            p = policy[0][self.move_hash[move.uci()]]
            candidates[move] = p
        x = sorted(candidates.items(), key=lambda x:x[1], reverse=True)
        
        print(x[0][1])
        print(v.item())
        
        return x[0][0]  #返回move


    def alpha_beta_search(self, board, depth, alpha, beta, color):
        '''
        board是当前棋面
        depth是当前搜索深度
        alpha beta
        color是当前的下棋者
        '''
        #如果搜到游戏结束了，直接返回INF
        if board.is_game_over():
            if board.is_stalemate():
                return 0
            return -color*self.INF
        
        #如果达到搜索层数，直接返回value
        if depth == 0:
            if color == 1:
                return -self.valuation(chess.Board(first_person_view_fen(self.board.fen(), 1)))
            else:
                return self.valuation(board)

        #在policy网络预测的值取前4个概率最大的走子
        legal_moves_list = list(board.legal_moves)
        #print()

        policy_list = []
        if color == 1:
            feature_plane = convert_board_to_plane(board.fen())
            feature_plane = feature_plane[np.newaxis, :]
            policy, _ = self.model.predict(feature_plane, batch_size=1)

            policy = [first_person_view_policy(policy[0], is_black_turn(board.fen()))]
            policy_list = [ (move, policy[0][self.move_hash[move.uci()]]) for move in legal_moves_list]
            policy_list = sorted(policy_list, key=lambda x:x[1], reverse=True) #从大到小排序
        else:
            for move in legal_moves_list:
                policy_list.append((move, 1))

        #搜索前4个
        threshold = min(0.01, max([policy_v[1] for policy_v in policy_list]))
        lim = 5

        for move, policy_value in (policy_list[:lim] if color == 1 else policy_list):
            if policy_value < threshold:
                continue

            board.push(move)

            value = self.alpha_beta_search(board, depth - 1, alpha, beta, -color)

            board.pop()
            if self.search_depth == depth:
                self.move_value[move] = value
                #print(move.uci())
            '''if depth == 1:
                print(move.uci(), is_black_turn(board.fen()), ' ', board.fen(), 'policy=', _)'''
            if color == 1:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if beta <= alpha:
                break
        return alpha if color == 1 else beta
    
    def valuation(self, board):
        #返回value network的估计值 (默认是针对白方)
        '''feature_plane = get_feature_plane(board.fen())
        feature_plane = feature_plane[np.newaxis, :]
        value = self.value_model.predict(feature_plane, batch_size=1)'''
        #print(board.fen(), ' ', value[0][0])
        value = evaluate_board(board.fen())
        return value


def convert_black_uci(move):
    new_move = []
    for s in move:
        if s.isdigit():
            new_move.append(str(9 - int(s)))
        else:
            new_move.append(s)
    return ''.join(new_move)

def count_piece(board_fen):
    '''
    count many pieces left on the board
    '''
    cnt = 0
    for pos in board_fen:
        if pos.isalpha():
            cnt += 1
    return cnt

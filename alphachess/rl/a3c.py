import time 
from collections import deque
import os
import random

import torch 
import torch.nn.functional as F 
import chess
import chess.pgn
import numpy as np


from alphachess.model import AlphaChess
from alphachess.utils import *


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, step_counter, game_counter, writer, lock, config , optimizer=None):
    '''
    己方永远执白
    '''
    torch.manual_seed(args.seed + rank)

    all_moves = get_all_possible_moves()
    move_hash = {move: i for (i, move) in enumerate(all_moves)}

    model = AlphaChess(config)
    oppo_model = AlphaChess(config)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    board = chess.Board()
    
    while True:
        board.reset()
        values = []
        log_probs = []
        rewards = []
        entropies = []

        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())

        # 从所有model中随机选一个
        model_list = os.listdir(config.resources.rl_model_dir)
        
        choise = int(random.random() * len(model_list))
        oppo_model.load_state_dict(torch.load(os.path.join(config.resources.rl_model_dir, model_list[choise]))['state_dict'])

        game = chess.pgn.Game()
        node = game

        # a new episode start!
        for step in range(args.num_steps):
            state = get_feature_plane(board.fen())
            state = state[np.newaxis, :].astype(np.float32)
            policy, v = model(torch.from_numpy(state))

            prob = F.softmax(policy, dim=1)
            log_prob = F.log_softmax(policy, dim=1)

            entropy = -(log_prob * prob).sum(1, keepdim=True)

            legal_moves = board.legal_moves
            legal_indices = [move_hash[move.uci()] for move in legal_moves]

            prob = prob.gather(1, torch.LongTensor([legal_indices]))
            prob = F.softmax(prob, dim=1) # 本来按理不需要这个。。。直到我遇到下一行报错
            action = legal_indices[prob.multinomial(1).item()]  #注意应该是采样，而不是取最大的
            log_prob = log_prob[0][action]

            action = board.parse_uci(all_moves[action])
            #print(action)
            board.push(action)
            #print(board)
            node = node.add_variation(action)

            values.append(v)
            log_probs.append(log_prob)
            entropies.append(entropy)
            
            if board.is_game_over():
                break
        
            
            # 对手的走棋 ,执黑
            oppo_board = chess.Board(first_person_view_fen(board.fen(), True))
            state = get_feature_plane(oppo_board.fen())
            state = state[np.newaxis, :].astype(np.float32)
            policy, v = oppo_model(torch.from_numpy(state))

            prob = F.softmax(policy, dim=1)
            
            legal_moves = oppo_board.legal_moves
            legal_indices = [move_hash[move.uci()] for move in legal_moves]
            prob = prob.gather(1, torch.LongTensor([legal_indices]))
            prob = F.softmax(prob, dim=1)
            action = legal_indices[prob.multinomial(1).item()]  #注意应该是采样，而不是取最大的

            action = board.parse_uci(first_person_view_move(all_moves[action], True))
            #print(action)
            board.push(action)
            #print(board)
            node = node.add_variation(action)

            if board.is_game_over():
                break

            rewards.append(0)   # 下的时候还是0
            
            with lock:
                step_counter.value += 1
            
        
        result = board.result()
        game.headers["Result"] = result

        if result == "1-0":
            rewards.append(1)
        elif result == "0-1":
            rewards.append(-1)
        elif result == "1/2-1/2":  # 和棋不进行反向传播
            continue
        else:
            # 看子多子少 很小的一个reward
            rewards[-1] = evaluate_board(board.fen()) / 10
        
        
        # 下完了，反向传播        
        R = torch.zeros(1, 1)
        if not board.is_game_over():
            state = get_feature_plane(board.fen())
            state = state[np.newaxis, :].astype(np.float32)
            _, v = model(torch.from_numpy(state))
            R = v
        
        values.append(R)
        gae = torch.zeros(1, 1)
        policy_loss = 0
        value_loss = 0

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            diff = R - values[i]
            value_loss += 0.5 * diff.pow(2)

            # Generalized Advantage Estimataion
            advantage = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.tau + advantage

            policy_loss -= log_probs[i] * gae   # 熵惩罚项

        ent_sum = 0.0
        for ent in entropies:
            ent_sum += ent
        
        
        print("data/a3c/policy_loss", policy_loss.item())
        print("data/a3c/value_loss", value_loss.item())
        print("data/a3c/entropy", ent_sum.item())
        
        policy_loss -=  args.entropy_coef * ent_sum

        optimizer.zero_grad()
        loss = policy_loss + args.value_loss_coef * value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        with lock:
            game_counter.value += 1
            print(game, file=open("data/self_play/{0}.pgn".format(game_counter.value), "w"), end="\n\n")
            # 每50迭代就更新一次模型
            if game_counter.value % 500 == 0:
                state = {"state_dict":shared_model.state_dict(),
                        "game_count":game_counter.value,
                        "step_num":step_counter.value}
                torch.save(state, "data/model/rl/alphachess_{0}.pth".format(game_counter.value))


def test(rank, args, shared_model,  step_counter, game_counter, writer, lock, config):
    start_time = time.time()
    while True:
        time.sleep(60)
        writer.add_scalar('data/a3c/fps', step_counter.value / (time.time() - start_time))
        writer.add_scalar('data/a3c/step_num', step_counter.value)
        writer.add_scalar('data/a3c/game_num', game_counter.value)
 
    writer.close()

import os

from lux.game import Game
from lux.game_map import RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
import numpy as np
import pandas as pd
import time

# TODO Añadir estados n/d y investigado o no
# TODO ¿Paralelizar?

DIRECTIONS = Constants.DIRECTIONS
game_state = None
unit_qtable = None
output_unit_qtable_file = 'out/qtable.csv'
save_ite = 1

actions_space = [Constants.DIRECTIONS.NORTH, Constants.DIRECTIONS.WEST, Constants.DIRECTIONS.SOUTH,
                 Constants.DIRECTIONS.EAST, Constants.ACTIONS.BUILD_CITY]
actions_cities = ['SpawnWorkerAction', 'SpawnCartAction', 'ResearchAction']
team_id = -1
last_action = {}


def agent(observation, configuration):
    global game_state, alpha, gamma, epsilon, unit_qtable, team_id, last_action
    # print('-------------------- Start Turn', observation["step"], '--------------------')
    ts_total = time.time()
    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player

        # Reinforcement Learning
        alpha = 0.1  # learning-rate
        gamma = 0.7  # discount-factor
        epsilon = 0  # explor vs exploit

        if os.path.isfile(output_unit_qtable_file):
            unit_qtable = pd.read_csv(output_unit_qtable_file, index_col=0)
        else:
            unit_qtable = pd.DataFrame(np.zeros([1, len(actions_space)]), columns=actions_space, index=['INIT'])
    else:
        game_state._update(observation["updates"])

    if is_day(game_state.turn) == 'n':
        ts = time.time()
        #unit_qtable.to_csv(output_unit_qtable_file, float_format='%.3f')
        unit_qtable.to_csv(output_unit_qtable_file)
        print('Save time', time.time() - ts)

    ### AI Code goes down here! ###
    player = game_state.players[observation.player]
    # print('Finish', len(player.units), len(player.cities.values()), observation["step"])
    team_id = player.team

    for unit_past in last_action:
        # print(unit_past, last_action[unit_past])
        update_unit_qtable(last_action[unit_past][0], last_action[unit_past][1], last_action[unit_past][2], player,
                           unit_past)

    actions = []
    rng = np.random.default_rng()
    for unit in player.units:
        if unit.is_worker() and unit.can_act():
            current_state = get_state(unit.pos.x, unit.pos.y, game_state.turn)
            if rng.random() < epsilon:
                current_action = select_unit_random_action(unit)
            else:
                if unit_qtable[unit_qtable.index == current_state].shape[0] == 1:
                    # Select randomly the action when max value is tied
                    current_action = unit_qtable[unit_qtable.index == current_state].idxmax().sample(1).index[0]
                else:
                    current_action = select_unit_random_action(unit)
                append_action = unit.build_city() if current_action == 'b' else unit.move(current_action)
            last_action[unit.id] = [current_action, current_state, unit.cargo_to_fuel()]
            actions.append(append_action)
        elif unit.is_cart():
            actions.append(unit.move(Constants.DIRECTIONS.CENTER))

    for city in player.cities.values():
        for city_tile in city.citytiles:
            if city_tile.can_act():
                # GetState
                if rng.random() < epsilon:
                    current_action, index_action = select_city_random_action(city_tile)
                    if index_action == 0:
                        actions.append(city_tile.build_worker())
                    elif index_action == 1:
                        actions.append(city_tile.build_cart())
                    else:
                        actions.append(city_tile.research())

    print('-------------------- End Turn', observation["step"], time.time() - ts_total, '--------------------')
    return actions


def select_city_random_action(city_tile):
    rng = np.random.default_rng()
    pos = int(rng.random() * len(actions_cities))
    current_action = actions_cities[pos]
    return current_action, pos


def update_unit_qtable(last_action, last_state, last_fuel_value, player, unit_id):
    global unit_qtable
    unit = player.find_unit_by_id(unit_id)
    # cuidado que puede haber muerto una unidad, de ahi el if
    if unit:
        reward = get_unit_reward(unit, last_fuel_value)
        next_state = get_state(unit.pos.x, unit.pos.y, game_state.turn)
        next_max = 0
        current_value = 0
        if unit_qtable.index.isin([next_state])[0]:
            next_max = unit_qtable.loc[next_state].max()

        if unit_qtable[unit_qtable.index == last_state].shape[0] == 1:
            current_value = unit_qtable.loc[last_state][last_action]
        else:
            new_row = pd.DataFrame(np.zeros([1, len(actions_space)]), columns=actions_space, index=[last_state])
            unit_qtable = pd.concat([unit_qtable, new_row])

        # Compute the new Q-value with the Bellman equation
        unit_qtable.loc[last_state, last_action] = (1 - alpha) * current_value + alpha * (reward + gamma * next_max)
    else:
        unit_qtable.loc[last_state, last_action] = alpha * gamma


def get_state(x, y, turn) -> str:
    return is_day(turn) + '' + \
           get_cell_value(x, y - 1) + '' + \
           get_cell_value(x - 1, y) + '' + \
           get_cell_value(x, y + 1) + '' + \
           get_cell_value(x + 1, y) + '' + \
           get_cell_value(x, y)


def select_unit_random_action(unit):
    rng = np.random.default_rng()
    if unit.can_build(game_state.map):
        pos = int(rng.random() * len(actions_space))
    else:
        pos = int(rng.random() * len(actions_space) - 1)
    if pos == len(actions_space) - 1:
        return Constants.ACTIONS.BUILD_CITY
    else:
        return actions_space[pos]


def get_cell_value(x, y) -> str:
    if 0 <= x < game_state.map.width and 0 <= y < game_state.map.height:
        cell = game_state.map.get_cell(x, y)
        if cell.has_resource():
            if cell.resource.type == RESOURCE_TYPES.WOOD:
                return RESOURCE_TYPES.WOOD[0]
            elif cell.resource.type == RESOURCE_TYPES.COAL:
                return RESOURCE_TYPES.COAL[0]
            elif cell.resource.type == RESOURCE_TYPES.URANIUM:
                return RESOURCE_TYPES.URANIUM[0]
        elif cell.citytile and cell.citytile.team == team_id:
            return 'b'
        elif cell.citytile and cell.citytile.team != team_id:
            return 'x'
        else:
            return '-'
    else:
        return '*'
    return '-'


def get_unit_reward(unit, last_fuel_value) -> int:
    if last_fuel_value < unit.cargo_to_fuel():
        return 1
    elif get_state(unit.pos.x, unit.pos.y, game_state.turn)[-1] == 'b':
        return 1
    else:
        return 0


def is_day(turn) -> bool:
    day_length = GAME_CONSTANTS['PARAMETERS']['DAY_LENGTH'] + GAME_CONSTANTS['PARAMETERS']['NIGHT_LENGTH']
    return 'd' if turn % day_length <= GAME_CONSTANTS['PARAMETERS']['DAY_LENGTH'] else 'n'

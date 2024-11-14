"""
Original author: Cam Allen
Modified from https://github.com/camall3n/pix2sym/tree/dev
"""
from enum import Enum
from collections import deque
from copy import deepcopy

import numpy as np

from environments.atarienv import actions


options = Enum('options', start=0, names=[
    'NONE',
    'RUN_LEFT',
    'RUN_RIGHT',
    'RUN_LEFT3',
    'RUN_RIGHT3',
    'JUMP_LEFT',
    'JUMP_RIGHT',
    'JUMP',
    'CLIMB_UP',
    'CLIMB_DOWN',
    'WAIT_FOR_SKULL',
    'WAIT_5',
    'WAIT_10',
    'WAIT_1',
    'STEP_RIGHT',
    'STEP_LEFT',
    'SAVE',
    'LOAD'
    ])

WAIT_50 = [options.WAIT_10, options.WAIT_10, options.WAIT_10, options.WAIT_10, options.WAIT_10]
ladders = Enum('ladders', start=0, names=['ANY', 'TOP', 'LEFT', 'RIGHT'])
enemy_detection_message = "Enemy spotted!\nTake evasive action!"


def makeWait(x):
    waits = []
    while x > 50:
        waits += WAIT_50
        x -= 50
    while x > 10:
        waits.append(options.WAIT_10)
        x -= 10
    while x > 1:
        waits.append(options.WAIT_1)
        x -= 1
    return waits


def find_nearest(array, value):
    # Find the element in 'array' that is the closest to 'value'
    idx = (np.abs(np.asarray(array) - value)).argmin()
    return array[idx]


def pixelsSame(a, b):
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


def isPixelBlack(pixel, frame):
    real_pixel = frame[pixel[0]][pixel[1]]
    return real_pixel[0] == 0 and real_pixel[1] == 0 and real_pixel[2] == 0


def allPixelsBlack(pixels):
    for row in range(len(pixels)):
        for cell in range(len(pixels[0])):
            if not isPixelBlack((row, cell), pixels):
                return False
    return True


def allPixelsSame(pixels):
    color = pixels[0, 0]
    for row in range(len(pixels)):
        for cell in range(len(pixels[0])):
            if not pixelsSame(color, pixels[row, cell, :]):
                return False
    return True


def borderNonBlackInnerBlack(pixels):
    # should only be used for ladder detection
    if not len(pixels[0]) == 10:
        return False
    if not len(pixels) == 7:
        return False
    borderColor = pixels[0][0]
    innerColor = pixels[1][1]

    if leq(innerColor, borderColor):
        return False
    for row in range(len(pixels)):
        for cell in range(len(pixels[0])):
            if row != 0 and row != len(pixels) - 1 and cell != 0 and cell != len(pixels[0]) - 1:
                if not leq(innerColor, pixels[row][cell]):
                    return False
            elif not leq(borderColor, pixels[row][cell]):
                return False
    return True


def leq(l1, l2):
    if len(l1) != len(l2):
        return False
    for elt in range(len(l1)):
        if not l1[elt] == l2[elt]:
            return False
    return True


def remove_columns(pixels):
    for col in range(0, len(pixels[0])):
        allOneColor = True
        for row in range(0, len(pixels)):
            if (not leq(pixels[0][col], pixels[row][col])):
                allOneColor = False
                break
        if (allOneColor):
            for row in range(0, len(pixels)):
                pixels[row][col] = [0, 0, 0]


def remove_rectangle(pixels, start):
    # start is the coordinates of a pixel that determines the color of the rectangle
    # by convention, this is the top left corner of the rectangle
    end_row = 0
    end_col = 0
    color = pixels[start[0]][start[1]]

    # next 2 loops identify the largest potential rectangle height and width
    for row in range(start[0], len(pixels)):
        end_row = row
        if not leq(color, pixels[row][start[1]]):
            end_row -= 1
            break
    for col in range(start[1], len(pixels[0])):
        end_col = col
        if not leq(color, pixels[start[0]][col]):
            end_col -= 1
            break

    # verify that we have an actual rectangle (based on shape)
    if (end_row == start[0] and end_col == start[1]) or \
       ((end_row == start[0] and not end_row == len(pixels) - 1) or
       (end_col == start[1] and not end_col == len(pixels[0]) - 1)):
        if not end_col == 0 or end_col == len(pixels[0]):
            return

    # Now, we want to verify that all pixels in our rectangular region are the
    # same color.
    for row in range(start[0], end_row + 1):
        for col in range(start[1], end_col + 1):
            if not leq(color, pixels[row][col]):
                # our rectangle wasn't uniform color, bail out
                return

    # we've found a rectangle we can remove, modify the pixel array to set the
    # rectangle pixels to black.
    for row in range(start[0], end_row + 1):
        for col in range(start[1], end_col + 1):
            pixels[row][col] = [0, 0, 0]


class SkillController:
    """Controller for hand-coded skills
    """
    def __init__(self, initial_plan=None, eps=0.0):
        """Initialize SkillController (optionally by specifying a plan)

        If provided, 'plan' should be a collections.deque of options
        """
        self.option = options.NONE
        self.frame = 0
        self.noop_max = 180
        self.noop_count = 0
        self.skillFunction = {
            options.NONE: self.noop,
            options.RUN_LEFT: self.runLeft,
            options.RUN_RIGHT: self.runRight,
            options.RUN_LEFT3: self.runLeft3,
            options.RUN_RIGHT3: self.runRight3,
            options.JUMP_LEFT: self.jumpLeft,
            options.JUMP_RIGHT: self.jumpRight,
            options.JUMP: self.jumpUp,
            options.CLIMB_UP: self.climbUp,
            options.CLIMB_DOWN: self.climbDown,
            options.WAIT_FOR_SKULL: self.waitForSkull,
            options.WAIT_1: self.wait1,
            options.WAIT_5: self.wait5,
            options.WAIT_10: self.wait10,
            options.STEP_RIGHT: self.stepRight,
            options.STEP_LEFT: self.stepLeft,
            options.SAVE: self.save,
            options.LOAD: self.load
        }
        self.nQueuedSkillsExecuted = 0
        self.plan = initial_plan
        self.eps = eps

        self.isInitialized = False
        self.lastX = deque([0] * 10)
        self.isActualRun = True
        self.wasWaiting = False
        self.reset()

    @property
    def running_plan(self):
        if self.initial_plan is None:
            return False
        return len(self.initial_plan) > 0

    def __call__(self, state):
        return self.runSkillPolicy(state)

    def __len__(self):
        return len(self.skillFunction)

    def reset(self):
        self.initialize_plan(self.plan, self.eps)

    def initialize_plan(self, initial_plan, eps):
        self.initial_plan = deepcopy(initial_plan)
        if initial_plan is not None:
            # with eps probability, remove some portion
            # of the perfect plan
            if np.random.rand() < eps:
                drop_count = np.random.randint(len(self.initial_plan))
                for _ in range(drop_count):
                    self.initial_plan.pop()
                # print(f"Updated the plan--new length={len(self.initial_plan)}")

    def getQueuedSkill(self, state):
        # Attempt to run the skill that is next in the queue
        try:
            o = self.initial_plan.popleft()
            (canExecute, _, _) = self.skillFunction[o](state)
            self.option = o
            # if canExecute:
            #     self.option = o
            # else:
            #     self.option = options.NONE
            #     raise ValueError
        except ValueError:
            print("Invaild skill.")
            raise
        except IndexError:
            print("Queue empty.")
            self.skillPrompt(state)
        # If no error was raised, increment counter
        self.nQueuedSkillsExecuted += 1
        return

    def getValidSkills(self, state):
        """Return the list of skills that are valid in the specified state

        This runs the skills' implicit classifiers
        """
        valid_ops = np.zeros(len(options))
        self.isActualRun = False
        for idx, op in enumerate(list(options)):
            (canExecute, _, _) = self.skillFunction[op](state)
            if canExecute:
                valid_ops[idx] = 1
        self.isActualRun = True
        return valid_ops

    def getRandomSkill(self, state):
        # Choose the next skill randomly from the valid skills in the specified state
        valid_ops = self.getValidSkills(state)
        op_id = (options.NONE.value +
                 np.random.choice(len(options), p=(valid_ops / np.sum(valid_ops))))
        self.option = options(op_id)

    def skillPrompt(self, state):
        # Ask user to specify a skill to run next
        while self.option == options.NONE:
            skill = input('''
    Select a skill:
    0. NONE
    1. RUN_LEFT
    2. RUN_RIGHT
    3. RUN_LEFT3
    4. RUN_RIGHT3
    5. JUMP_LEFT
    6. JUMP_RIGHT
    7. JUMP
    8. CLIMB_UP
    9. CLIMB_DOWN
    10. WAIT_FOR_SKULL
    11. WAIT_5
    12. WAIT_10
    13. WAIT_1
    14. STEP_RIGHT
    15. STEP_LEFT
    > ''')
            try:
                skill = int(skill)
                if skill in range(16):
                    print("You selected {}.".format(options(skill).name))
                    o = options(skill)
                    (canExecute, _, _) = self.skillFunction[o](state)
                    if canExecute:
                        self.option = o
                    else:
                        raise ValueError
            except ValueError:
                print("Invaild skill.")

    def isRunningSkill(self):
        if self.option != options.NONE:
            return True
        else:
            return False

    def chooseNextSkill(self, state):
        if self.initial_plan:
            try:
                self.getQueuedSkill(state)
            except ValueError:
                if self.nQueuedSkillsExecuted > 0:
                    self.initial_plan = None
                    self.getRandomSkill(state)
        else:
            self.isInitialized = True
            self.getRandomSkill(state)
        self.frame = 0

    def didSkillTimeout(self, action, op_valid):
        timeout = False
        if self.noop_count >= self.noop_max:
            self.noop_count = 0
            timeout = True
        elif not op_valid and action == actions.NOOP:
            # Count number of NOOPs in a row if the current skill is invalid
            self.noop_count += 1
        ###
        # added to prevent some skill deadlocks
        elif not self.running_plan:
            self.noop_count += 1
        ###
        else:
            self.noop_count = 0
        return timeout

    def runSkillPolicy(self, state):
        """Run the current skill (or if it has completed, run the next skill)

        Returns a tuple consisting of (action, option, frame, op_done)
            action - the next low-level action
            option - the option that is currently running
            op_frame - the number of frames since the current option started
            op_done - whether this frame is the last one for this option
        """
        if not self.isRunningSkill():
            self.chooseNextSkill(state)

        skill_fn = self.skillFunction[self.option]
        (op_valid, action, op_done) = skill_fn(state)
        timeout = self.didSkillTimeout(action, op_valid)
        if not op_done and (state['respawned'] or timeout):
            action = actions.NOOP
            op_done = True

        action_frame_tuple = (action, self.option, self.frame, op_done)
        self.frame += 1
        if op_done:
            ###
            # added to prevent some skill deadlocks
            if not self.running_plan:
                self.noop_count = 0
            ###
            self.option = options.NONE
            self.frame = 0
        return action_frame_tuple

    def noop(self, state):
        canExecute = True  # unless another option can run
        for o in list(options)[1:]:
            (valid, _, _) = self.skillFunction[o](state)
            if valid:
                canExecute = False
                break
        action = actions.NOOP
        lastFrame = True
        return (canExecute, action, lastFrame)

    def save(self, state):
        if self.isActualRun:
            print("Type saveawd")
            filePath = "saves/" + input()
            state['env'].save(filePath)
        return self.noop(state)

    def load(self, state):
        if self.isActualRun:
            print("Type loading")
            filePath = "saves/" + input()
            state['env'].load(filePath)
        return self.noop(state)

    def runRight(self, state):
        return self.run(state, actions.RIGHT)

    def runLeft(self, state):
        return self.run(state, actions.LEFT)

    def runRight3(self, state):
        return self.run(state, actions.RIGHT, True)

    def runLeft3(self, state):
        return self.run(state, actions.LEFT, True)

    def wait1(self, state):
        return self.wait(state, 1)

    def wait5(self, state):
        return self.wait(state, 5)

    def wait10(self, state):
        return self.wait(state, 10)

    def wait(self, state, times):
        if self.isActualRun:
            if self.wasWaiting:
                remainingTimes = times - self.waitedAlready
            else:
                self.wasWaiting = True
                remainingTimes = times
                self.waitedAlready = 0
            if remainingTimes < 2:
                self.wasWaiting = False
            self.waitedAlready += 1
            return (True, actions.NOOP, not self.wasWaiting)
        else:
            return (True, actions.NOOP, False)

    def stepRight(self, state):
        return self.step(state, 1, actions.RIGHT)

    def stepLeft(self, state):
        return self.step(state, 1, actions.LEFT)

    def step(self, state, times, direction):
        if self.isActualRun:
            if self.wasWaiting:
                remainingTimes = times - self.waitedAlready
            else:
                self.wasWaiting = True
                remainingTimes = times
                self.waitedAlready = 0
            if remainingTimes == 1:
                self.wasWaiting = False
            self.waitedAlready += 1
            return (True, direction, not self.wasWaiting)
        else:
            return (True, direction, False)

    def run(self, state, direction, is3=False):
        x = state['player_x']
        if self.isActualRun:
            self.lastX.append(x)
            self.lastX.popleft()

        def thar_be_enemy(rgb, lbound, rbound):
            cut = rgb[31:40, lbound:rbound]
            cut = np.copy(cut)
            remove_columns(cut)
            for y in range(len(cut)):
                alll = True
                none = True
                for x in range(len(cut[0])):
                    if not isPixelBlack((y, x), cut):
                        remove_rectangle(cut, (y, x))
                    alll = alll and isPixelBlack((y, x), cut)
                    none = none and not (isPixelBlack((y, x), cut))
                if not (none or alll):
                    return True
            return False

        if self.isActualRun:
            rgb = state[f'pixels_around_player_{direction}']

        on_ground = not (state['player_falling'] or state['player_jumping']) and (
            state['player_status'] in ['standing', 'running']) and not state['just_died']
        if not on_ground:
            return (False, direction, False)
        # Potentially add one more block of padding if running off the edge
        stop_condition = False
        if self.isActualRun:
            alll = True
            for prevX in self.lastX:
                if not x == prevX:
                    alll = False
            if alll:
                self.lastX = deque([0] * 10)
            stop_condition = stop_condition or alll

        if direction == actions.LEFT:
            # enemy stop_condition
            if self.isActualRun:
                if is3:
                    stop_condition = stop_condition or thar_be_enemy(rgb, 7, 15)
                else:
                    stop_condition = stop_condition or thar_be_enemy(rgb, 10, 15)

                # platform edge stop_condition
                cut = rgb[45:53, -6:, :]
                if len(cut[0]) > 0:
                    stop_condition = stop_condition or (allPixelsBlack(cut[2:6, 0:1, :]))

                # Ladder detection below and at level
                cut = rgb[45:51, -6:, :]
                if len(cut) == 7 and len(cut[0]) == 10:
                    stop_condition = stop_condition or borderNonBlackInnerBlack(cut)

                for i in range(12, 55):
                    cut = rgb[i:i + 7, 18:, :]
                    stop_condition = stop_condition or borderNonBlackInnerBlack(cut)
                    cut = rgb[i:i + 7, 17:-1, :]
                    stop_condition = stop_condition or borderNonBlackInnerBlack(cut)
            return (True, actions.LEFT, stop_condition)
        if direction == actions.RIGHT:
            # enemy stop_condition
            if self.isActualRun:
                if is3:
                    stop_condition = stop_condition or thar_be_enemy(rgb, 12, 20)
                else:
                    stop_condition = stop_condition or thar_be_enemy(rgb, 12, 17)

                # platform edge detection
                cut = rgb[45:53, :7, :]
                stop_condition = stop_condition or (allPixelsBlack(cut[2:6, 6:7, :]))

                # Ladder detection
                for i in range(43, 55):
                    cut = rgb[i:i + 7, 1:11, :]
                    if len(cut) == 7 and len(cut[0]) == 10:
                        stop_condition = stop_condition or borderNonBlackInnerBlack(cut)

                # This is wrong, fix it
                cut = rgb[17:24, 2:12, :]
                stop_condition = stop_condition or borderNonBlackInnerBlack(cut)
            return (True, actions.RIGHT, stop_condition)
        return (True, actions.LEFT, False)

    def jumpUp(self, state):
        return self.jump(state, actions.FIRE)

    def jumpRight(self, state):
        return self.jump(state, actions.RIGHT_FIRE)

    def jumpLeft(self, state):
        return self.jump(state, actions.LEFT_FIRE)

    def jump(self, state, direction):
        canExecute = False
        action = actions.NOOP
        lastFrame = False

        on_ground = (
            state['player_status'] in ['standing', 'running']
            and not (state['player_jumping'] or state['player_falling'] or state['just_died']))

        on_rope = (state['player_status'] in ['on-rope', 'climbing-rope']
                   and not (state['player_jumping'] or state['player_falling']))

        if on_ground or (on_rope and direction != actions.FIRE):
            # Assume we have yet to jump
            canExecute = True
            action = direction
            if (self.frame > 4
                    and self.option in [options.JUMP, options.JUMP_LEFT, options.JUMP_RIGHT]):
                # Override if we've already jumped + landed
                lastFrame = True
                action = actions.NOOP

        return (canExecute, action, lastFrame)

    def climbUp(self, state):
        return self.climb(state, actions.UP)

    def climbDown(self, state):
        return self.climb(state, actions.DOWN)

    def climb(self, state, direction):
        canExecute = False
        action = direction
        lastFrame = False

        rgb = state[f'pixels_around_player_{direction}']
        ladder_below = False
        for i in range(46, 60):
            cut = rgb[i:i + 7, 12:22, :]
            if borderNonBlackInnerBlack(cut):
                ladder_below = True
                break
            cut = rgb[i:i + 7, 13:23, :]
            if borderNonBlackInnerBlack(cut):
                ladder_below = True
                break

        at_floor = allPixelsBlack(rgb[46:47, 8:11, :]) and allPixelsBlack(rgb[46:47, 21:24, :])
        at_floor = (at_floor or allPixelsBlack(rgb[47:48, 8:11, :])
                    and allPixelsBlack(rgb[47:48, 21:24, :]))
        at_floor = (at_floor or allPixelsSame(rgb[45:46, 2:28, :])
                    or allPixelsSame(rgb[46:47, 2:28, :]))

        lastFrame = at_floor

        can_climb = (
            not (state['just_died'] or state['player_jumping'] or state['player_falling'])
            and state['player_status'] in [
                'standing', 'running', 'on-ladder', 'climbing-ladder', 'on-rope', 'climbing-rope']
        )

        if can_climb and ladder_below:
            canExecute = True

        return (canExecute, action, lastFrame)

    def render(self, state):
        return (True, actions.NOOP, True)

    def waitForSkull(self, state):
        def thar_be_enemy(rgb, lbound, rbound):
            cut = rgb[31:40, lbound:rbound]
            cut = np.copy(cut)
            remove_columns(cut)
            for y in range(len(cut)):
                alll = True
                none = True
                for x in range(len(cut[0])):
                    if not isPixelBlack((y, x), cut):
                        remove_rectangle(cut, (y, x))
                    alll = alll and isPixelBlack((y, x), cut)
                    none = none and not (isPixelBlack((y, x), cut))
                if not (none or alll):
                    return True
            return False

        lastFrame = False
        rgb = state[f'pixels_around_player_{actions.RIGHT}']
        lastFrame = thar_be_enemy(rgb, 10, 15) or thar_be_enemy(rgb, 12, 17)

        # the perfect plan runs with a different logic
        # where joe jumps from the rope to the platform
        # when skull and joe have different y
        if self.running_plan:
            is_valid = True
        # i believe this should be the sensible precondition
        # for this skill
        else:
            is_valid = False
            if state['skull_y'] == state['player_y']:
                is_valid = True

        return (is_valid, actions.NOOP, lastFrame)


class Plans:
    """Class for conveniently specifying an initial plan to a SkillController
    """
    # https://www.youtube.com/watch?v=sYbBgkP9aMo

    # Movement
    S0_TO_NW = [options.RUN_LEFT, options.JUMP_LEFT]
    S0_TO_NE = [options.RUN_RIGHT, options.JUMP_RIGHT]
    S0_TO_MID = [options.CLIMB_DOWN, options.CLIMB_DOWN]
    S0_TO_E = S0_TO_MID + [options.RUN_RIGHT, options.JUMP_RIGHT, options.JUMP_RIGHT]
    S0_TO_SE = S0_TO_E + [options.CLIMB_DOWN, options.CLIMB_DOWN, options.RUN_LEFT]
    S0_TO_SW = S0_TO_SE + [options.JUMP_LEFT, options.WAIT_FOR_SKULL, options.JUMP_LEFT, options.RUN_LEFT]
    S0_TO_W = S0_TO_SW + [options.RUN_LEFT, options.CLIMB_UP, options.CLIMB_UP]
    S0_TO_KEY = S0_TO_W + [options.JUMP_LEFT]

    KEY_TO_W = [options.RUN_RIGHT]
    KEY_TO_SW = KEY_TO_W + [options.CLIMB_DOWN, options.CLIMB_DOWN, options.RUN_RIGHT]
    KEY_TO_SE = KEY_TO_SW + [options.WAIT_FOR_SKULL, options.JUMP_RIGHT, options.RUN_RIGHT]
    KEY_TO_E = KEY_TO_SE + [options.RUN_RIGHT, options.CLIMB_UP, options.CLIMB_UP]
    KEY_TO_MID = KEY_TO_E + [options.JUMP_LEFT, options.JUMP_LEFT, options.RUN_LEFT]
    KEY_TO_S0 = KEY_TO_MID + [options.CLIMB_UP]
    KEY_TO_NW = KEY_TO_S0 + S0_TO_NW
    KEY_TO_NE = KEY_TO_S0 + S0_TO_NE

    S0_TO_SKULL = S0_TO_SE + [options.RUN_LEFT]
    KEY_TO_SKULL = KEY_TO_SW + [options.RUN_RIGHT]
    S0_TO_KEY_NOSKULL = S0_TO_SE + [options.RUN_LEFT, options.RUN_LEFT, options.CLIMB_UP, options.JUMP_LEFT]

    # Base positions
    S0 = []
    NE = S0_TO_NE
    NW = S0_TO_NW
    MID = S0_TO_MID
    E = S0_TO_E
    SE = S0_TO_SE
    SW = S0_TO_SW
    W = S0_TO_W

    # Positions w/ key
    S0_KEY = S0_TO_KEY + KEY_TO_S0
    NW_KEY = S0_TO_KEY + KEY_TO_NW
    NE_KEY = S0_TO_KEY + KEY_TO_NE
    MID_KEY = S0_TO_KEY + KEY_TO_MID
    E_KEY = S0_TO_KEY + KEY_TO_E
    SE_KEY = S0_TO_KEY + KEY_TO_SE
    SW_KEY = S0_TO_KEY + KEY_TO_SW
    W_KEY = S0_TO_KEY + KEY_TO_W

    # Positions w/o skull
    S0_SKULL = S0_TO_SKULL
    NW_SKULL = S0_TO_SKULL + S0_TO_NW
    NE_SKULL = S0_TO_SKULL + S0_TO_NE
    MID_SKULL = S0_TO_SKULL + S0_TO_MID
    E_SKULL = S0_TO_SKULL + S0_TO_E
    SE_SKULL = S0_TO_SKULL + S0_TO_SE
    SW_SKULL = SE_SKULL + [options.RUN_LEFT]
    W_SKULL = SW_SKULL + [options.RUN_LEFT, options.CLIMB_UP]

    # Positions w/ key + w/o skull
    S0_BOTH = S0_TO_KEY + KEY_TO_SKULL
    NE_BOTH = S0_TO_KEY + KEY_TO_SKULL + S0_TO_NE
    NW_BOTH = S0_TO_KEY + KEY_TO_SKULL + S0_TO_NW
    MID_BOTH = S0_TO_KEY + KEY_TO_SKULL + S0_TO_MID
    E_BOTH = S0_TO_KEY + KEY_TO_SKULL + S0_TO_E
    SE_BOTH = S0_TO_KEY + KEY_TO_SKULL + S0_TO_SE
    SW_BOTH = S0_TO_SKULL + S0_TO_KEY_NOSKULL + KEY_TO_SW
    W_BOTH = S0_TO_SKULL + S0_TO_KEY_NOSKULL + KEY_TO_W

    # Intentionally lose a life. Lose N lives with (S0_TO_DIE * N)
    S0_TO_DIE = [options.RUN_RIGHT, options.RUN_RIGHT]

    JUMP_5_TIMES = [options.JUMP] * 5
    WAIT = [options.NONE] * 5
    WAIT_FOR_SKULL = [options.WAIT_FOR_SKULL]

    RUN_LEFT = [options.RUN_LEFT]
    RUN_RIGHT = [options.RUN_RIGHT]
    RUN_LEFT3 = [options.RUN_LEFT3]
    RUN_RIGHT3 = [options.RUN_RIGHT3]
    JUMP_LEFT = [options.JUMP_LEFT]
    JUMP_RIGHT = [options.JUMP_RIGHT]
    JUMP = [options.JUMP]
    DOWN = [options.CLIMB_DOWN, options.CLIMB_DOWN]
    UP = [options.CLIMB_UP, options.CLIMB_UP]

    SMALL_STEP_RIGHT = [options.STEP_RIGHT]
    STEP_RIGHT = (SMALL_STEP_RIGHT + SMALL_STEP_RIGHT + SMALL_STEP_RIGHT + SMALL_STEP_RIGHT +
                  SMALL_STEP_RIGHT + SMALL_STEP_RIGHT + SMALL_STEP_RIGHT)
    MIDDLE_STEP_RIGHT = SMALL_STEP_RIGHT + SMALL_STEP_RIGHT + SMALL_STEP_RIGHT
    SMALL_STEP_LEFT = [options.STEP_LEFT]
    STEP_LEFT = (SMALL_STEP_LEFT + SMALL_STEP_LEFT + SMALL_STEP_LEFT + SMALL_STEP_LEFT +
                 SMALL_STEP_LEFT + SMALL_STEP_LEFT + SMALL_STEP_LEFT)
    MIDDLE_STEP_LEFT = SMALL_STEP_LEFT + SMALL_STEP_LEFT + SMALL_STEP_LEFT

    SAVE = [options.SAVE]
    LOAD = [options.LOAD]

    FORCE_LEFT = [options.RUN_LEFT, options.RUN_LEFT, options.RUN_LEFT]
    FORCE_UP = UP + UP + UP + UP + UP + UP + UP + UP + UP + UP
    FORCE_DOWN = DOWN + DOWN + DOWN + DOWN + DOWN + DOWN + DOWN + DOWN + DOWN + DOWN

    BARRIER_PASS_LEFT = makeWait(110) + RUN_LEFT
    BARRIER_PASS_RIGHT = makeWait(110) + RUN_RIGHT

    R1_START_SKULL = [
        options.CLIMB_DOWN, options.CLIMB_DOWN, options.RUN_RIGHT, options.JUMP_RIGHT,
        options.JUMP_RIGHT, options.RUN_RIGHT, options.CLIMB_DOWN, options.CLIMB_DOWN,
        options.RUN_LEFT, options.JUMP_LEFT
    ]
    R1_SKULL_KEY = [options.RUN_LEFT, options.CLIMB_UP, options.CLIMB_UP, options.RUN_LEFT, options.JUMP]
    R1_KEY_SKULL = [options.RUN_RIGHT, options.CLIMB_DOWN, options.CLIMB_DOWN, options.RUN_RIGHT, options.JUMP_RIGHT]
    R1_SKULL_START = [
        options.RUN_RIGHT, options.CLIMB_UP, options.CLIMB_UP, options.RUN_LEFT,
        options.JUMP_LEFT, options.JUMP_LEFT, options.RUN_LEFT, options.CLIMB_UP,
        options.CLIMB_UP
    ]
    R1_START_RIGHT = [options.RUN_RIGHT, options.JUMP_RIGHT, options.RUN_RIGHT, options.JUMP_RIGHT, options.RUN_RIGHT]
    R1_START_LEFT = [options.RUN_LEFT, options.JUMP_LEFT, options.RUN_LEFT, options.JUMP_LEFT, options.RUN_LEFT]

    # NOTE: Rooms in this plan are 1-indexed, but rooms in the state are 0-indexed!
    ROOM_2 = R1_START_SKULL + R1_SKULL_KEY + R1_KEY_SKULL + R1_SKULL_START + R1_START_LEFT
    ROOM_1 = (RUN_LEFT + RUN_LEFT + BARRIER_PASS_LEFT + BARRIER_PASS_LEFT + RUN_LEFT + RUN_LEFT +
              makeWait(60) + RUN_LEFT + [options.JUMP] + makeWait(75) + RUN_RIGHT + DOWN)
    ROOM_5 = makeWait(100) + DOWN + RUN_LEFT + makeWait(230) + RUN_LEFT + JUMP_LEFT + RUN_LEFT
    ROOM_4 = makeWait(40) + RUN_LEFT + RUN_LEFT + DOWN
    ROOM_10 = (DOWN + DOWN + RUN_LEFT + RUN_LEFT + RUN_LEFT + JUMP_LEFT + RUN_LEFT + RUN_LEFT +
               RUN_LEFT + JUMP_LEFT + RUN_LEFT)
    ROOM_9 = (RUN_LEFT + JUMP_LEFT + FORCE_UP + UP + UP + JUMP + FORCE_DOWN + FORCE_DOWN +
              FORCE_DOWN + FORCE_DOWN + RUN_RIGHT + makeWait(50) + RUN_RIGHT + JUMP + JUMP + JUMP +
              makeWait(75) + JUMP_5_TIMES + RUN_RIGHT)
    ROOM_10_B = (STEP_RIGHT + JUMP_RIGHT + STEP_RIGHT + STEP_RIGHT + JUMP_RIGHT + STEP_RIGHT +
                 STEP_RIGHT + STEP_RIGHT + UP)
    ROOM_4_B = UP + RUN_RIGHT
    ROOM_5_B = RUN_RIGHT + RUN_RIGHT + JUMP_RIGHT + makeWait(5) + RUN_LEFT + DOWN
    ROOM_11 = (DOWN + makeWait(100) + DOWN + RUN_LEFT + JUMP_RIGHT + RUN_RIGHT + STEP_RIGHT +
               STEP_RIGHT + UP + DOWN + DOWN + makeWait(60) + DOWN + RUN_RIGHT)
    ROOM_12 = RUN_RIGHT + MIDDLE_STEP_RIGHT + JUMP_RIGHT + RUN_RIGHT + RUN_RIGHT + UP
    ROOM_6 = (UP + RUN_RIGHT + JUMP_RIGHT + STEP_RIGHT + MIDDLE_STEP_RIGHT + STEP_RIGHT + JUMP +
              FORCE_UP + JUMP_LEFT + RUN_LEFT + JUMP_LEFT + RUN_LEFT + JUMP_LEFT + UP + UP + UP +
              MIDDLE_STEP_RIGHT + JUMP_RIGHT + JUMP_RIGHT + STEP_RIGHT + STEP_RIGHT + STEP_RIGHT +
              STEP_RIGHT + JUMP_RIGHT + FORCE_DOWN + JUMP_LEFT + RUN_LEFT + DOWN)
    ROOM_12_B = DOWN + RUN_RIGHT + MIDDLE_STEP_RIGHT + JUMP_RIGHT + RUN_RIGHT
    ROOM_13 = (makeWait(20) + RUN_RIGHT + makeWait(75) + RUN_RIGHT + makeWait(110) + RUN_RIGHT +
               makeWait(115) + RUN_RIGHT + makeWait(110) + RUN_RIGHT)
    ROOM_14 = makeWait(100) + RUN_RIGHT + RUN_RIGHT + JUMP_RIGHT + RUN_RIGHT + UP + UP
    ROOM_8 = makeWait(130) + RUN_LEFT + makeWait(40) + RUN_LEFT + makeWait(110) + RUN_LEFT
    ROOM_7 = (RUN_LEFT + RUN_LEFT + STEP_LEFT + STEP_LEFT + RUN_LEFT + RUN_RIGHT + JUMP_RIGHT +
              RUN_RIGHT + STEP_RIGHT + STEP_RIGHT + RUN_RIGHT)
    ROOM_8_B = (makeWait(65) + RUN_RIGHT + BARRIER_PASS_RIGHT + RUN_RIGHT + makeWait(60) +
                RUN_RIGHT + JUMP + makeWait(80) + RUN_LEFT + DOWN)
    ROOM_14_B = DOWN + STEP_RIGHT + JUMP_RIGHT + JUMP_RIGHT + RUN_RIGHT
    ROOM_15 = (RUN_RIGHT + RUN_RIGHT + JUMP_RIGHT + JUMP_RIGHT + JUMP_LEFT + FORCE_DOWN +
               FORCE_DOWN + FORCE_DOWN + FORCE_DOWN + RUN_RIGHT + DOWN + DOWN)
    ROOM_23 = makeWait(70) + DOWN + RUN_RIGHT
    ROOM_24 = RUN_RIGHT + RUN_RIGHT + STEP_LEFT + JUMP_LEFT + STEP_LEFT + JUMP_LEFT + RUN_LEFT
    ROOM_23_B = (makeWait(120) + RUN_LEFT + [options.CLIMB_UP] + makeWait(100) + DOWN + RUN_LEFT +
                 MIDDLE_STEP_LEFT + JUMP_LEFT + RUN_LEFT)
    ROOM_22 = makeWait(60) + RUN_LEFT + JUMP_LEFT + UP
    ROOM_14_C = DOWN + RUN_LEFT
    ROOM_13_B = (makeWait(130) + RUN_LEFT + makeWait(20) + RUN_LEFT + makeWait(120) + RUN_LEFT +
                 makeWait(110) + RUN_LEFT + makeWait(110) + RUN_LEFT)
    ROOM_12_C = RUN_LEFT + RUN_LEFT + MIDDLE_STEP_LEFT + JUMP_LEFT + RUN_LEFT + DOWN + DOWN
    ROOM_20 = DOWN + RUN_RIGHT
    ROOM_21 = (makeWait(70) + RUN_RIGHT + JUMP_RIGHT + STEP_RIGHT + JUMP_RIGHT + RUN_RIGHT +
               makeWait(100) + JUMP_LEFT + JUMP_LEFT + RUN_LEFT)
    ROOM_20_B = (RUN_LEFT + RUN_LEFT + STEP_LEFT + STEP_LEFT + RUN_LEFT + RUN_RIGHT + JUMP_RIGHT +
                 RUN_LEFT)
    TO_END = (makeWait(40) + RUN_LEFT + JUMP_LEFT + RUN_LEFT + JUMP_LEFT + RUN_LEFT + JUMP_LEFT +
              RUN_LEFT + JUMP_LEFT + RUN_LEFT + RUN_LEFT + RUN_LEFT)

    END_A = (JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT + JUMP_LEFT + JUMP +
             JUMP_RIGHT + JUMP_LEFT + JUMP_RIGHT + JUMP_LEFT + JUMP_LEFT)
    END_B = JUMP_LEFT + JUMP_LEFT + JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT
    END = END_A + END_B

    BROOM_2 = ROOM_2
    BROOM_1 = (makeWait(30) + RUN_LEFT + makeWait(110) + RUN_LEFT + JUMP_LEFT + RUN_LEFT +
               makeWait(50) + RUN_LEFT + JUMP + makeWait(80) + RUN_RIGHT + DOWN)
    BROOM_5 = DOWN + RUN_LEFT + MIDDLE_STEP_LEFT + JUMP_LEFT + RUN_LEFT
    BROOM_4 = (RUN_LEFT + RUN_LEFT + JUMP_LEFT + RUN_LEFT + JUMP_RIGHT + JUMP_RIGHT + RUN_RIGHT +
               DOWN)
    BROOM_10 = DOWN + DOWN + RUN_LEFT + JUMP_LEFT
    BROOM_9 = (RUN_LEFT + RUN_LEFT + MIDDLE_STEP_RIGHT + JUMP_LEFT + FORCE_DOWN + FORCE_DOWN +
               RUN_LEFT + JUMP_LEFT + makeWait(60) + JUMP + JUMP + JUMP + makeWait(60) +
               JUMP_5_TIMES + RUN_RIGHT + MIDDLE_STEP_LEFT + SMALL_STEP_LEFT + JUMP_RIGHT +
               FORCE_DOWN + RUN_RIGHT + JUMP_RIGHT + JUMP + JUMP + JUMP + makeWait(30) +
               JUMP_5_TIMES + RUN_RIGHT)
    BROOM_10_B = RUN_RIGHT + RUN_RIGHT + JUMP_RIGHT + RUN_RIGHT + STEP_RIGHT + STEP_RIGHT + UP
    BROOM_4_B = UP + RUN_RIGHT
    BROOM_5_B = RUN_RIGHT + RUN_RIGHT + JUMP_RIGHT + RUN_RIGHT + STEP_RIGHT + STEP_RIGHT + DOWN
    BROOM_11 = (DOWN + DOWN + STEP_LEFT + STEP_LEFT + STEP_LEFT + STEP_LEFT + JUMP_LEFT +
                JUMP_LEFT + RUN_RIGHT + STEP_RIGHT + STEP_RIGHT + [options.CLIMB_UP] + makeWait(70) +
                DOWN + RUN_RIGHT)
    BROOM_12 = RUN_RIGHT + STEP_RIGHT + STEP_RIGHT + UP
    BROOM_6 = ROOM_6
    BROOM_12_B = DOWN + RUN_RIGHT
    BROOM_13 = (makeWait(20) + RUN_RIGHT + RUN_RIGHT + makeWait(105) + RUN_RIGHT +
                BARRIER_PASS_RIGHT + makeWait(10) + BARRIER_PASS_RIGHT + BARRIER_PASS_RIGHT)
    BROOM_14 = RUN_RIGHT + RUN_RIGHT + JUMP_RIGHT + RUN_RIGHT + STEP_RIGHT + STEP_RIGHT + UP
    BROOM_8 = (UP + makeWait(20) + RUN_RIGHT + makeWait(110) + RUN_RIGHT + JUMP + makeWait(75) +
               RUN_LEFT + RUN_LEFT + makeWait(70) + RUN_LEFT + BARRIER_PASS_LEFT)
    BROOM_7 = RUN_LEFT + RUN_LEFT + UP
    BROOM_3 = UP + RUN_RIGHT + JUMP_LEFT + RUN_LEFT + DOWN
    BROOM_7_B = DOWN + DOWN + RUN_RIGHT
    BROOM_8_B = makeWait(90) + RUN_RIGHT + BARRIER_PASS_RIGHT + DOWN
    BROOM_14_B = DOWN + RUN_RIGHT + JUMP_RIGHT + RUN_RIGHT
    BROOM_15 = (RUN_RIGHT + RUN_RIGHT + JUMP_RIGHT + FORCE_UP + FORCE_UP + STEP_RIGHT + JUMP +
                STEP_LEFT + FORCE_DOWN + FORCE_DOWN + FORCE_DOWN + FORCE_DOWN + RUN_RIGHT + DOWN)
    BROOM_23 = DOWN + DOWN + RUN_RIGHT
    BROOM_24 = RUN_RIGHT + RUN_RIGHT + JUMP_LEFT + JUMP_LEFT + RUN_LEFT
    BROOM_23_B = (makeWait(80) + RUN_LEFT + JUMP_LEFT + JUMP_LEFT + JUMP_LEFT + JUMP_LEFT +
                  JUMP_LEFT + RUN_LEFT)
    BROOM_22 = (RUN_LEFT + RUN_LEFT + JUMP_LEFT + RUN_LEFT + JUMP_RIGHT + RUN_RIGHT + STEP_RIGHT +
                STEP_RIGHT + UP)
    BROOM_14_C = UP + DOWN + RUN_LEFT + MIDDLE_STEP_LEFT + JUMP_LEFT + RUN_LEFT
    BROOM_13_B = (RUN_RIGHT + makeWait(20) + RUN_LEFT + makeWait(110) + RUN_LEFT + makeWait(110) +
                  RUN_LEFT + makeWait(115) + RUN_LEFT + makeWait(110) + RUN_LEFT)
    BROOM_12_C = RUN_LEFT + RUN_LEFT + DOWN + DOWN
    BROOM_20 = DOWN + RUN_RIGHT
    BROOM_21 = (makeWait(60) + RUN_RIGHT + JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT +
                JUMP_RIGHT + JUMP_LEFT + RUN_LEFT)
    BROOM_20_B = RUN_LEFT + RUN_LEFT + UP
    BROOM_12_D = (UP + DOWN + RUN_LEFT + RUN_RIGHT + JUMP_RIGHT + RUN_RIGHT + STEP_RIGHT +
                  STEP_RIGHT + DOWN)
    BROOM_20_C = DOWN + DOWN + RUN_LEFT
    BROOM_19 = makeWait(40) + JUMP_LEFT + JUMP_LEFT + RUN_LEFT + JUMP_LEFT + RUN_LEFT
    TO_BEND = RUN_LEFT + JUMP_LEFT + RUN_LEFT + JUMP_LEFT + RUN_LEFT + RUN_LEFT + RUN_LEFT
    BEND = (JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT +
            JUMP_LEFT + JUMP_LEFT + JUMP_LEFT + JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT +
            JUMP_LEFT + JUMP_LEFT + JUMP + JUMP + JUMP_RIGHT + JUMP + JUMP)

    CROOM_2 = (DOWN + RUN_RIGHT + JUMP_RIGHT + JUMP_RIGHT + RUN_RIGHT + DOWN + makeWait(30) +
               RUN_LEFT3 + JUMP_LEFT + RUN_LEFT + UP + RUN_LEFT + JUMP + RUN_RIGHT + DOWN +
               makeWait(10) + RUN_RIGHT3 + JUMP_RIGHT + RUN_RIGHT + UP + RUN_LEFT + JUMP_LEFT +
               JUMP_LEFT + RUN_LEFT + UP + RUN_LEFT + JUMP_LEFT + RUN_LEFT + JUMP_LEFT + RUN_LEFT)
    CROOM_1 = RUN_LEFT + makeWait(280) + RUN_LEFT + makeWait(105) + RUN_LEFT + DOWN
    CROOM_4 = (JUMP_LEFT + JUMP_LEFT + JUMP_LEFT + JUMP_LEFT + JUMP_LEFT + JUMP_LEFT + RUN_RIGHT +
               DOWN)
    CROOM_5 = (DOWN + RUN_LEFT + MIDDLE_STEP_LEFT + JUMP_LEFT + RUN_LEFT + MIDDLE_STEP_LEFT +
               JUMP_LEFT + RUN_LEFT)
    CROOM_10 = (DOWN + DOWN + DOWN + JUMP_LEFT + MIDDLE_STEP_LEFT + MIDDLE_STEP_LEFT +
                MIDDLE_STEP_LEFT + makeWait(10) + JUMP_LEFT + RUN_LEFT)
    CROOM_9 = (RUN_LEFT + RUN_LEFT + STEP_RIGHT + JUMP_LEFT + FORCE_DOWN + RUN_LEFT + JUMP_LEFT +
               JUMP + JUMP + JUMP + makeWait(40) + JUMP_5_TIMES + JUMP_RIGHT + RUN_RIGHT +
               STEP_LEFT + JUMP_RIGHT + DOWN + DOWN + RUN_RIGHT + JUMP_RIGHT + JUMP + JUMP + JUMP +
               makeWait(40) + JUMP_5_TIMES + RUN_RIGHT)
    CROOM_10_B = (MIDDLE_STEP_RIGHT + MIDDLE_STEP_RIGHT + MIDDLE_STEP_RIGHT + JUMP_RIGHT +
                  RUN_RIGHT + STEP_RIGHT + STEP_RIGHT + UP)
    CROOM_4_B = UP + RUN_RIGHT
    CROOM_5_B = (STEP_RIGHT + JUMP_RIGHT + RUN_RIGHT + JUMP_RIGHT + RUN_RIGHT + STEP_RIGHT +
                 STEP_RIGHT + DOWN)
    CROOM_11 = (makeWait(91) + DOWN + RUN_LEFT + JUMP_RIGHT + RUN_RIGHT + JUMP_RIGHT + RUN_RIGHT +
                JUMP_RIGHT + STEP_RIGHT + STEP_RIGHT + STEP_RIGHT + JUMP_RIGHT + RUN_RIGHT)
    CROOM_12 = makeWait(45) + RUN_RIGHT + RUN_RIGHT + UP
    CROOM_6 = (UP + RUN_RIGHT + JUMP_RIGHT + RUN_RIGHT + JUMP_LEFT + JUMP_LEFT + FORCE_UP +
               makeWait(30) + JUMP_LEFT + JUMP_LEFT + STEP_LEFT + JUMP_LEFT + RUN_LEFT +
               JUMP_LEFT + UP + UP + UP + MIDDLE_STEP_RIGHT + JUMP_RIGHT + JUMP_RIGHT +
               STEP_RIGHT + STEP_RIGHT + STEP_RIGHT + STEP_RIGHT + JUMP_RIGHT + FORCE_DOWN +
               JUMP_LEFT + RUN_LEFT + DOWN)
    CROOM_12_B = DOWN + DOWN
    CROOM_20_A = DOWN + DOWN + makeWait(80) + UP
    CROOM_12_BB = DOWN + RUN_RIGHT
    CROOM_13 = (RUN_RIGHT + RUN_RIGHT + makeWait(60) + RUN_RIGHT + makeWait(110) + RUN_RIGHT +
                makeWait(120) + RUN_RIGHT + makeWait(110) + RUN_RIGHT)
    CROOM_14 = RUN_RIGHT + RUN_RIGHT + UP
    CROOM_8 = UP + makeWait(130) + RUN_RIGHT + JUMP + makeWait(75) + RUN_LEFT + DOWN
    CROOM_14_B = DOWN + RUN_RIGHT
    CROOM_15 = ROOM_15
    CROOM_23 = makeWait(90) + DOWN + RUN_LEFT
    CROOM_22 = RUN_LEFT + RUN_LEFT + JUMP_LEFT + RUN_LEFT + UP
    CROOM_14_C = UP + DOWN + RUN_LEFT
    CROOM_13_B = (RUN_LEFT + makeWait(80) + RUN_LEFT + makeWait(115) + RUN_LEFT + makeWait(115) +
                  RUN_LEFT + makeWait(110) + RUN_LEFT)
    CROOM_12_C = makeWait(30) + RUN_LEFT + RUN_LEFT + DOWN
    CROOM_20 = DOWN + DOWN + RUN_RIGHT
    CROOM_21 = (makeWait(50) + RUN_RIGHT + JUMP_RIGHT + RUN_RIGHT + JUMP_LEFT + makeWait(90) +
                JUMP_LEFT + RUN_LEFT)
    CROOM_20_B = RUN_LEFT + RUN_LEFT + JUMP_LEFT + RUN_LEFT
    CROOM_19 = (makeWait(90) + JUMP_LEFT + JUMP_LEFT + JUMP_LEFT + JUMP_LEFT + JUMP_LEFT +
                MIDDLE_STEP_LEFT + MIDDLE_STEP_LEFT + JUMP_LEFT + RUN_LEFT)
    TO_CEND = RUN_LEFT + JUMP_LEFT + RUN_LEFT + JUMP_LEFT + RUN_LEFT + RUN_LEFT + RUN_LEFT
    CEND = (JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT +
            JUMP_RIGHT + JUMP_LEFT + JUMP_LEFT + JUMP_LEFT + JUMP_RIGHT + JUMP_LEFT + JUMP_RIGHT +
            JUMP + JUMP_LEFT + JUMP_RIGHT + JUMP_RIGHT + JUMP_RIGHT + JUMP_LEFT + JUMP_LEFT)

    LEVEL_1 = (ROOM_2 + ROOM_1 + ROOM_5 + ROOM_4 + ROOM_10 + ROOM_9 + ROOM_10_B + ROOM_4_B +
               ROOM_5_B + ROOM_11 + ROOM_12 + ROOM_6 + ROOM_12_B + ROOM_13 + ROOM_14 + ROOM_8 +
               ROOM_7 + ROOM_8_B + ROOM_14_B + ROOM_15 + ROOM_23 + ROOM_24 + ROOM_23_B + ROOM_22 +
               ROOM_14_C + ROOM_13_B + ROOM_12_C + ROOM_20 + ROOM_21 + ROOM_20_B + TO_END + END)

    LEVEL_2 = (BROOM_2 + BROOM_1 + BROOM_5 + BROOM_4 + BROOM_10 + BROOM_9 + BROOM_10_B +
               BROOM_4_B + BROOM_5_B + BROOM_11 + BROOM_12 + BROOM_6 + BROOM_12_B + BROOM_13 +
               BROOM_14 + BROOM_8 + BROOM_7 + BROOM_3 + BROOM_7_B + BROOM_8_B + BROOM_14_B +
               BROOM_15 + BROOM_23 + BROOM_24 + BROOM_23_B + BROOM_22 + BROOM_14_C + BROOM_13_B +
               BROOM_12_C + BROOM_20 + BROOM_21 + BROOM_20_B + BROOM_12_D + BROOM_20_C + BROOM_19 +
               TO_BEND + BEND)

    LEVEL_3 = (CROOM_2 + CROOM_1 + CROOM_5 + CROOM_4 + CROOM_10 + CROOM_9 + CROOM_10_B +
               CROOM_4_B + CROOM_5_B + CROOM_11 + CROOM_12 + CROOM_6 + CROOM_12_B + CROOM_20_A +
               CROOM_12_BB + CROOM_13 + CROOM_14 + CROOM_8 + CROOM_14_B + CROOM_15 + CROOM_23 +
               CROOM_22 + CROOM_14_C + CROOM_13_B + CROOM_12_C + CROOM_20 + CROOM_21 + CROOM_20_B +
               CROOM_19 + TO_CEND + CEND)

    FULL_RUN = LEVEL_1 + LEVEL_2 + LEVEL_3
    FROM_SAVE = LOAD + CEND + SAVE

    safe_starts = [
        S0, NW, NE, MID, E, SE, SW, W, S0_KEY, NW_KEY, NE_KEY, MID_KEY, E_KEY, SE_KEY, SW_KEY,
        W_KEY
    ]
    death_starts = [
        S0_SKULL,
        NW_SKULL,
        NE_SKULL,
        MID_SKULL,
        E_SKULL,
        SE_SKULL,
        SW_SKULL,
        W_SKULL,
        S0_BOTH,
        NW_BOTH,
        NE_BOTH,
        MID_BOTH,
        E_BOTH,
        SE_BOTH,
        SW_BOTH,
        W_BOTH,
    ]
    all_starts = safe_starts + death_starts

    base_plan = [options.NONE]

    n_lives = 5
    n_starts = n_lives * len(all_starts) + len(safe_starts)

    @classmethod
    def GetFullRunPlan(cls):
        """Choose the hand-coded full run plan

        Returns the plan as a collections.deque
        """
        return deque(cls.FULL_RUN)

    @classmethod
    def GetStartByIndex(cls, i):
        """Choose the initial plan associated with index 'i'

        Returns the plan as a collections.deque
        """
        assert i >= 0 and i < cls.n_starts
        i %= len(cls.all_starts)
        p = cls.base_plan
        p += cls.all_starts[i]
        return deque(p)

    @classmethod
    def RandomStart(cls):
        """Choose a random initial plan and return it as a collections.deque
        """
        i = np.random.randint(0, Plans.n_starts)
        return cls.GetStartByIndex(i)

    @classmethod
    def GetDefaultStart(cls):
        return cls.GetStartByIndex(0)

    @classmethod
    def WrapPlanSteps(cls, plan_steps):
        return deque(cls.base_plan + plan_steps)

    @classmethod
    def ToRoomNumber(cls, room_number):
        # NOTE: room_number argument is 0-indexed
        # but the implementation is 1-indexed for legacy reasons
        translated_room_number = room_number + 1

        # Here and below, (parentheses) denote multiple room plans that we combine together, since
        # some don't reach a new room on their own. We only care about reaching specific rooms in
        # this function.
        rooms_with_existing_plans = [
            2, 1, 5, 4, 10, 9,
            (11),
            12, 6,
            (13),
            14, 8, 7,
            (15),
            23, 24,
            (22),
            20,
            (21),
            19,
            16
        ]
        partial_plan_to_subsequent_rooms = [
            [],
            cls.ROOM_2, cls.ROOM_1, cls.ROOM_5, cls.ROOM_4, cls.ROOM_10,
            (cls.ROOM_9 + cls.ROOM_10_B + cls.ROOM_4_B + cls.ROOM_5_B),
            cls.ROOM_11, cls.ROOM_12,
            (cls.ROOM_6 + cls.ROOM_12_B),
            cls.ROOM_13, cls.ROOM_14, cls.ROOM_8,
            (cls.ROOM_7 + cls.ROOM_8_B + cls.ROOM_14_B),
            cls.ROOM_15, cls.ROOM_23,
            cls.ROOM_24 + cls.ROOM_23_B,
            (cls.ROOM_22 + cls.ROOM_14_C + cls.ROOM_13_B + cls.ROOM_12_C),
            cls.ROOM_20,
            (cls.ROOM_21 + cls.ROOM_20_B),
            cls.TO_END,
            []
        ]

        if translated_room_number in rooms_with_existing_plans:
            idx = rooms_with_existing_plans.index(translated_room_number)
            plan_fragments = partial_plan_to_subsequent_rooms[:idx + 1]
            combined_plan = sum(plan_fragments, [])
            return deque(combined_plan)

        raise NotImplementedError('Room {} does not have an existing plan'.format(room_number))

    @classmethod
    def ToSpiderRoom(cls):
        return cls.ToRoomNumber(4)

    @classmethod
    def ToSpiderRoomFloor(cls):
        plan = cls.ToRoomNumber(4)
        [plan.extend(item) for item in [makeWait(100) + cls.DOWN]]
        return plan

    @classmethod
    def ToJumpSkullRoom(cls):
        return cls.ToRoomNumber(3)

    @classmethod
    def ToJumpSkullRoomLadder(cls):
        plan = cls.ToRoomNumber(3)
        plan += makeWait(40) + cls.RUN_LEFT + cls.RUN_LEFT + cls.DOWN + [options.CLIMB_UP] * 2
        return plan

    @classmethod
    def ToJumpSkullRoomLadder2(cls):
        plan = deque([options.NONE] + Plans.NE_KEY + [options.JUMP_RIGHT] * 4)
        return plan

    @classmethod
    def ToStartingRoom(cls):
        return deque([options.NONE] * 5)

    @classmethod
    def ToStartingRoomBottom(cls):
        return deque(cls.S0_TO_SE) + deque(cls.RUN_RIGHT)

    @classmethod
    def ToLaserRoom(cls, location='laser_screen-0_right'):

        # Room 0
        s0_r = cls.ToRoomNumber(0)
        s0_l = cls.ToRoomNumber(0)
        for item in [makeWait(110) + cls.RUN_LEFT]:
            s0_l.extend(item)
        s0_m = cls.ToRoomNumber(0)
        for item in [makeWait(110) + cls.RUN_LEFT + makeWait(110) + cls.RUN_LEFT]:
            s0_m.extend(item)

        # Room 7
        s7_m = cls.ToRoomNumber(7)
        for item in [cls.UP]:
            s7_m.extend(item)

        # Room 12
        s12_l = cls.ToRoomNumber(12)
        s12_r = cls.ToRoomNumber(12)
        for item in [makeWait(110) + cls.RUN_RIGHT]:
            s12_r.extend(item)

        valid_locations = {
            'laser_screen-0_right': s0_r,
            'laser_screen-0_left': s0_l,
            'laser_screen-0_middle': s0_m,
            'laser_screen-7_middle': s7_m,
            'laser_screen-12_right': s12_r,
            'laser_screen-12_left': s12_l,
        }
        if location not in valid_locations:
            raise ValueError("No such location {} in platform room".format(location))
        return valid_locations[location]

    @classmethod
    def ToPlatformRoom(cls, location='enter'):

        ENTER_ROOM = cls.ToRoomNumber(8)
        TOP_RIGHT_EDGE = ENTER_ROOM + deque(cls.RUN_LEFT)
        ROPE = TOP_RIGHT_EDGE + deque(cls.JUMP_LEFT)
        TOP_CENTER = ROPE + deque(cls.FORCE_UP + cls.UP + cls.UP + cls.JUMP)
        BOTTOM_CENTER = TOP_CENTER + deque(cls.FORCE_DOWN + cls.FORCE_DOWN + cls.FORCE_DOWN +
                                           cls.FORCE_DOWN)
        BOTTOM_LEFT = BOTTOM_CENTER + deque(cls.RUN_LEFT + makeWait(50) + cls.RUN_LEFT)
        BOTTOM_RIGHT = BOTTOM_CENTER + deque(cls.RUN_RIGHT + makeWait(50) + cls.RUN_RIGHT)
        TOP_LEFT = BOTTOM_LEFT + deque(cls.JUMP + cls.JUMP + cls.JUMP + makeWait(75) +
                                       cls.JUMP_5_TIMES)
        TOP_LEFT_EDGE = TOP_LEFT + deque(cls.RUN_RIGHT)
        TOP_RIGHT = BOTTOM_RIGHT + deque(cls.JUMP + cls.JUMP + cls.JUMP + makeWait(75) +
                                         cls.JUMP_5_TIMES)
        valid_locations = {
            'enter': ENTER_ROOM,
            'top-right-edge': TOP_RIGHT_EDGE,
            'rope': ROPE,
            'top-center': TOP_CENTER,
            'bottom-center': BOTTOM_CENTER,
            'bottom-left': BOTTOM_LEFT,
            'bottom-right': BOTTOM_RIGHT,
            'top-left': TOP_LEFT,
            # below are unused for now
            'top-left-edge': TOP_LEFT_EDGE,
            'top-right': TOP_RIGHT,
        }
        if location not in valid_locations:
            raise ValueError("No such location {} in platform room".format(location))
        return valid_locations[location]

    @classmethod
    def ToTorchRoom(cls):
        plan = cls.ToRoomNumber(5)
        plan.extend([options.CLIMB_UP, options.CLIMB_UP, options.CLIMB_UP])
        return plan

    @classmethod
    def ToBridgeRoom(cls):
        return cls.ToRoomNumber(10)

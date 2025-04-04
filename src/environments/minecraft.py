from PIL import Image
from typing import Optional
from collections import defaultdict

import minedojo
from minedojo.sim.mc_meta.mc import ALL_CRAFT_SMELT_ITEMS
import gym
import numpy as np
import torch

from s2s.helpers import dict_to_transition
from s2s.structs import UnorderedDataset

USEFUL_ITEMS = [
    "air",
    "log",
    "cobblestone",
    "crafting_table",
    "planks",
    "stick",
    "diamond_axe",
    "diamond_pickaxe",
    "furnace",
    "iron_axe",
    "iron_pickaxe",
    "stone_axe",
    "stone_pickaxe",
    "wooden_axe",
    "wooden_pickaxe",
    "diamond",
    "iron_ingot",
    "iron_ore",
]

ATTACK_ITEMS = [
    "wooden_axe",
    "wooden_pickaxe",
    "stone_axe",
    "stone_pickaxe",
    "iron_axe",
    "iron_pickaxe",
    "diamond_axe",
    "diamond_pickaxe"
]

PRIVILEGED_MAPPING = {
    "null": 0,
    "air": 1,
    "wood": 2,
    "cobblestone": 3,
    "wooden planks": 4,
    "crafting table": 5,
    "iron ore": 6,
    "diamond ore": 7,
    "furnace": 8,
}


class Minecraft(gym.Env):
    PLAN = [
        ("teleport", ("east",), ((-6, 4, 3),)),
        ("attack", ("air",), ()),
        ("teleport", ("east",), ((-6, 4, 6),)),
        ("attack", ("air",), ()),
        ("craft", ("planks",), ()),
        ("craft", ("planks",), ()),
        ("craft", ("stick",), ()),
        ("teleport", ("east",), ((0, 4, 0),)),
        ("craft", ("wooden_pickaxe",), ()),
        ("teleport", ("east",), ((3, 4, 3),)),
        ("attack", ("wooden_pickaxe",), ()),
        ("teleport", ("east",), ((3, 4, 6),)),
        ("attack", ("wooden_pickaxe",), ()),
        ("teleport", ("east",), ((3, 4, 9),)),
        ("attack", ("wooden_pickaxe",), ()),
        ("teleport", ("east",), ((0, 4, 0),)),
        ("craft", ("stone_pickaxe",), ()),
        ("teleport", ("east",), ((3, 4, -3),)),
        ("attack", ("stone_pickaxe",), ()),
        ("teleport", ("east",), ((3, 4, -6),)),
        ("attack", ("stone_pickaxe",), ()),
        ("teleport", ("east",), ((3, 4, -9),)),
        ("attack", ("stone_pickaxe",), ()),
        ("teleport", ("east",), ((-6, 4, 9),)),
        ("attack", ("air",), ()),
        ("craft", ("planks",), ()),
        ("teleport", ("east",), ((0, 4, 3),)),
        ("craft", ("iron_ingot",), ()),
        ("craft", ("iron_ingot",), ()),
        ("craft", ("iron_ingot",), ()),
        ("teleport", ("east",), ((0, 4, 0),)),
        ("craft", ("stick",), ()),
        ("craft", ("iron_pickaxe",), ()),
        ("teleport", ("east",), ((-3, 4, -3),)),
        ("attack", ("iron_pickaxe",), ()),
        ("teleport", ("east",), ((-3, 4, -6),)),
        ("attack", ("iron_pickaxe",), ()),
        ("teleport", ("east",), ((-3, 4, -9),)),
        ("attack", ("iron_pickaxe",), ()),
        ("teleport", ("east",), ((0, 4, 0),)),
        ("craft", ("diamond_pickaxe",), ()),
    ]

    def __init__(self, world_config: dict, world_seed: int = 0, seed: int = 0, max_steps=200):
        drawing_str = None
        initial_inventory = None
        world_limit = {"xmin": -10, "xmax": 10, "ymin": 4, "ymax": 10, "zmin": -10, "zmax": 10}
        if "blocks" in world_config:
            drawing_str = Minecraft.blocks_to_xml(world_config["blocks"])
        if "inventory" in world_config:
            initial_inventory = Minecraft.inventory_to_items(world_config["inventory"])
        if "world_limit" in world_config:
            world_limit = world_config["world_limit"]
        self._world_limit = world_limit
        self._initial_blocks = []
        for b in world_config["blocks"]:
            pos = b["position"]
            self._initial_blocks.append((int(pos["x"]), int(pos["y"]), int(pos["z"])))

        self._max_steps = max_steps
        self._t = 0
        if np.random.rand() < 0.5:
            self._plan_eps_idx = np.random.randint(0, len(self.PLAN))
        else:
            self._plan_eps_idx = len(self.PLAN) - 1

        self._env = minedojo.make(
            task_id="open-ended",
            generate_world_type="flat",
            image_size=(600, 800),
            world_seed=world_seed,
            seed=seed,
            drawing_str=drawing_str,
            initial_inventory=initial_inventory,
            start_position=dict(x=0.5, y=4, z=0.5, yaw=0, pitch=0),
            break_speed_multiplier=1000,
            allow_mob_spawn=False,
            use_lidar=True,
            lidar_rays=[(0, 0, 999)],
            allow_time_passage=False
        )

        self._block_map = {}
        self._agent_img = None
        self._last_targeted_block = None
        self._prev_obs = {}
        self._executed_actions = {}

    @property
    def observation(self) -> dict:
        obs = {}
        if self._agent_img is not None:
            img = self._agent_img[0].copy().reshape(-1)
            obs["agent"] = {0: img}
        else:
            obs["agent"] = {0: None}

        obs["inventory"] = {}
        inv = self.prev_obs["inventory"]
        for i in range(36):
            obs["inventory"][i] = np.array([
                USEFUL_ITEMS.index(inv["name"][i].replace(" ", "_")),
                int(inv["quantity"][i]),
                inv["variant"][i]
            ])

        obs["objects"] = {}
        for key in self._block_map:
            if self._block_map[key][0]:
                img = self._block_map[key][1].copy().reshape(-1)
                x, y, z = key
                is_agent_near = 0
                if (x+1, y, z) == self.agent_pos:
                    is_agent_near = 1
                is_agent_near = np.array([is_agent_near], dtype=np.uint8)
                obs["objects"][key] = np.concatenate([img, is_agent_near])
        obs["global"] = {0: np.array(self.agent_pos + (self.agent_dir,))}
        obs["dimensions"] = {"inventory": 3, "agent": 32*32*3, "objects": 32*32*3+1, "global": 4}
        return obs

    @property
    def reward(self) -> float:
        return float(self._has_diamond_pickaxe())

    @property
    def done(self) -> bool:
        if self._has_diamond_pickaxe():
            return True
        available_actions = self.available_actions()
        return (len(available_actions) == 0) or (self._t >= self._max_steps)

    def _has_diamond_pickaxe(self) -> bool:
        inv = self.prev_obs["inventory"]
        for i in range(36):
            if inv["name"][i] == "diamond pickaxe":
                return True
        return False

    @property
    def info(self) -> dict:
        obs = {}
        if self._agent_img is not None:
            block_name = self._agent_img[1]
            obs["agent"] = {0: np.array([PRIVILEGED_MAPPING[block_name], 0, 0, 0, 0, 0])}
        else:
            obs["agent"] = {0: np.array([0, 0, 0, 0, 0, 0])}

        obs["inventory"] = {}
        inv = self.prev_obs["inventory"]
        for i in range(9):
            obs["inventory"][i] = np.array([
                0, 0,
                USEFUL_ITEMS.index(inv["name"][i].replace(" ", "_")),
                int(inv["quantity"][i]),
                0, 0
            ])

        obs["objects"] = {}
        for key in self._block_map:
            if self._block_map[key][0]:
                block = PRIVILEGED_MAPPING[self._block_map[key][2]]
                x, y, z = key
                is_agent_near = 0
                if (x+1, y, z) == self.agent_pos:
                    is_agent_near = 1
                is_agent_near = np.array([is_agent_near], dtype=np.uint8)
                obs["objects"][key] = np.concatenate([[0, 0, 0, 0, block], is_agent_near])
        obs["global"] = {0: np.array(self.agent_pos + (self.agent_dir,))}
        obs["dimensions"] = {"agent": 6, "inventory": 6, "objects": 6, "global": 4}
        return obs

    @property
    def world_limit(self) -> dict:
        return self._world_limit

    @property
    def prev_obs(self) -> dict:
        return self._prev_obs

    @property
    def object_centric(self) -> bool:
        return True

    @property
    def agent_pos(self) -> tuple[int, int, int]:
        pos = self.prev_obs["location_stats"]["pos"]
        return int(np.floor(pos[0])), int(np.floor(pos[1])), int(np.floor(pos[2]))

    @property
    def agent_dir(self) -> int:
        # 0: east, 1: south, 2: west, 3: north
        yaw = self.prev_obs["location_stats"]["yaw"]
        if -135 < yaw <= -45:
            return 0
        elif -45 < yaw <= 45:
            return 1
        elif 45 < yaw <= 135:
            return 2
        else:
            return 3

    @property
    def last_targeted_block(self) -> Optional[tuple[int, int, int]]:
        return self._last_targeted_block

    @property
    def traced_block(self) -> tuple[int, int, int]:
        x = int(np.floor(self.prev_obs["rays"]["traced_block_x"]))
        y = int(np.floor(self.prev_obs["rays"]["traced_block_y"]))
        z = int(np.floor(self.prev_obs["rays"]["traced_block_z"]))
        return x, y, z

    @property
    def item_in_hand(self) -> str:
        return self.prev_obs["inventory"]["name"][0]

    def reset(self) -> dict:
        self._block_map = {}
        obs = self._env.reset()
        self._prev_obs = obs
        self._observe_all_blocks()
        self._t = 0
        if np.random.rand() < 0.5:
            self._plan_eps_idx = np.random.randint(0, len(self.PLAN))
        else:
            self._plan_eps_idx = len(self.PLAN) - 1
        return self.observation

    def step(self, action) -> tuple[dict, float, bool, dict]:
        action_type, action_args, object_args = action

        if (action_type == "teleport"):
            x, y, z = object_args[0]
            side = action_args[0]
            res = self._teleport_to_block(x, y, z, side)
        elif action_type == "attack":
            item_name = action_args[0]
            res = self._attack_block(item_name)
        elif action_type == "place":
            block_name = action_args[0]
            res = self._place_block(block_name)
        elif action_type == "craft":
            item_name = action_args[0]
            res = self._craft_item(item_name)
        else:
            raise ValueError("Invalid action type")

        if res:
            action_key = (action_type, action_args)
            if action_key not in self._executed_actions:
                self._executed_actions[action_key] = 0
            self._executed_actions[action_key] += 1
            self._t += 1

        obs = self.observation
        reward = self.reward
        done = self.done
        info = self.info
        info["action_success"] = res
        return obs, reward, done, info

    def sample_action(self):
        if self._t <= self._plan_eps_idx:
            return self.PLAN[self._t]
        return self._get_random_action()

    def available_actions(self):
        actions = []

        # possible teleport locations
        # teleport(block, side)
        for (x, y, z) in self._block_map:
            if not self._block_map[(x, y, z)][0]:
                continue
            actions.append(("teleport", ("east",), ((x, y, z),)))

        # attack_<with_item>()
        if self.last_targeted_block is not None:
            # do not attack crafting table and furnace
            if (self._agent_img[1] != "crafting table") and (self._agent_img[1] != "furnace"):
                for item_type in self._attack_items():
                    actions.append(("attack", (item_type,), ()))

        # craft_<craftable_item>()
        for item_type in self._craftable_items():
            if item_type not in USEFUL_ITEMS:
                continue
            if (item_type == "crafting_table") or (item_type == "furnace"):
                continue
            actions.append(("craft", (item_type,), ()))
        return actions

    def close(self):
        self._env.close()

    def _step(self, action):
        obs, _, _, _ = self._env.step(action)
        for _ in range(5):
            obs, _, _, _ = self._env.step(self.NO_OP())
        self._prev_obs = obs

    def _teleport_to_block(self, x: int, y: int, z: int, side: str) -> bool:
        if side == "east":
            self._teleport(x+1, y, z)
        elif side == "south":
            self._teleport(x, y, z+1)
        elif side == "west":
            self._teleport(x-1, y, z)
        elif side == "north":
            self._teleport(x, y, z-1)
        self._look_and_update_belief(x, y, z)
        return True

    def _attack_block(self, item_name: str) -> bool:
        x, y, z = self.last_targeted_block
        self._look_at_block(x, y, z)
        if self.traced_block != (x, y, z):
            return False
        exists = self._block_exists(x, y, z)
        if exists:
            if item_name != "air":
                self._equip_item(item_name)
            attack = Minecraft.ATTACK()
            self._step(attack)
            self._env.teleport_agent(x, y, z, 0, 0)
            self._wait(2)
            self._clear_belief(x, y, z)
        else:
            raise ValueError("This should never happen?!")
        return True

    def _place_block(self, item_name: str) -> bool:
        inventory_slot = self._get_item_index(item_name)
        ax, ay, az = self.agent_pos
        x, y, z = self.traced_block
        dx = ax - x
        dz = az - z
        target = (ax+dx, ay, az+dz)
        # if there is a block in the teleport location, skip
        if target in self._block_map and self._block_map[target][0]:
            return False

        self._teleport(ax+dx, ay, az+dz)
        self._look_at_block(x, y, z, noisy=False)

        place_idx = Minecraft.PLACE(inventory_slot)
        self._step(place_idx)
        self._look_and_update_belief(ax, ay, az)

        if (self._world_limit["xmin"] < ax) and ((ax-1, ay, az) not in self._block_map):
            self._block_map[(ax-1, ay, az)] = (False, "air", "air")
        if (self._world_limit["xmax"] > ax) and ((ax+1, ay, az) not in self._block_map):
            self._block_map[(ax+1, ay, az)] = (False, "air", "air")
        if (self._world_limit["zmin"] < az) and ((ax, ay, az-1) not in self._block_map):
            self._block_map[(ax, ay, az-1)] = (False, "air", "air")
        if (self._world_limit["zmax"] > az) and ((ax, ay, az+1) not in self._block_map):
            self._block_map[(ax, ay, az+1)] = (False, "air", "air")
        return True

    def _equip_item(self, item_name: str) -> bool:
        inventory_slot = self._get_item_index(item_name)
        equip = Minecraft.EQUIP(inventory_slot)
        self._step(equip)
        return True

    def _craft_item(self, item_name: str) -> bool:
        item_id = ALL_CRAFT_SMELT_ITEMS.index(item_name)
        craft = Minecraft.CRAFT(item_id)
        self._step(craft)
        return True

    def _get_random_action(self):
        actions = self.available_actions()
        action_types = defaultdict(list)
        for a in actions:
            action_types[a[0]].append(a)
        types = list(action_types.keys())
        a_t = np.random.choice(types)
        action_args = action_types[a_t]
        idx = np.random.choice(len(action_args))
        return action_args[idx]

    def _get_item_index(self, item_name: str) -> Optional[int]:
        indices, = np.where(self.prev_obs["inventory"]["name"] == item_name.replace("_", " "))
        if len(indices) == 0:
            return None
        return indices[0]

    def _attack_items(self):
        equip_mask = self.prev_obs["masks"]["equip"]
        item_names = np.unique(self.prev_obs["inventory"]["name"][equip_mask])
        item_names = [name.replace(" ", "_") for name in item_names]
        item_names = [name for name in item_names if name in ATTACK_ITEMS]
        item_names.append("air")
        return item_names

    def _placeable_items(self):
        place_mask = self.prev_obs["masks"]["place"]
        item_names = np.unique(self.prev_obs["inventory"]["name"][place_mask])
        item_names = [name.replace(" ", "_") for name in item_names]
        return item_names

    def _craftable_items(self):
        craft_smelt_mask, = np.where(self.prev_obs["masks"]["craft_smelt"])
        item_names = [ALL_CRAFT_SMELT_ITEMS[idx] for idx in craft_smelt_mask]
        return item_names

    def _wait(self, t):
        for _ in range(t):
            self._step(self._env.action_space.no_op())

    def _below_support_exists(self, x: int, y: int, z: int) -> bool:
        if y == self._world_limit["ymin"]:
            return True
        if (x, y-1, z) not in self._block_map:
            return False
        return self._block_map[(x, y-1, z)][0]

    def _observe_all_blocks(self) -> None:
        # This will be called only at the beginning of
        # each episode to scan the map and get an initial
        # observation of the blocks.
        # - Assuming we don't expect any transportation collision.
        # - Assuming the neighboring blocks are 'air'.
        for (x_i, y_i, z_i) in self._initial_blocks:
            self._teleport(x_i+1, y_i, z_i)
            self._look_and_update_belief(x_i, y_i, z_i)
            if self._world_limit["xmin"] < x_i:
                self._block_map[(x_i-1, y_i, z_i)] = (False, "air", "air")
            if self._world_limit["xmax"] > x_i:
                self._block_map[(x_i+1, y_i, z_i)] = (False, "air", "air")
            if self._world_limit["zmin"] < z_i:
                self._block_map[(x_i, y_i, z_i-1)] = (False, "air", "air")
            if self._world_limit["zmax"] > z_i:
                self._block_map[(x_i, y_i, z_i+1)] = (False, "air", "air")

        # place the crafting table
        self._teleport_to_block(-1, 4, 0, "east")
        self._place_block("crafting_table")
        self._teleport_to_block(-1, 4, 0, "west")
        self._attack_block("air")
        # and the furnace
        self._teleport_to_block(-1, 4, 3, "east")
        self._place_block("furnace")
        self._teleport_to_block(-1, 4, 3, "west")
        self._attack_block("air")

    def _look_and_update_belief(self, x: int, y: int, z: int) -> None:
        self._look_at_block(x, y, z, noisy=True)
        rays = self.prev_obs["rays"]
        if self.traced_block == (x, y, z):
            block_name = rays["block_name"][0]
        elif self._block_exists(x, y, z):
            block_name = self._block_map[(x, y, z)][2]
        else:
            block_name = "null"
        img = self._get_block_from_current_view(x, y, z)
        self._last_targeted_block = (x, y, z)
        self._block_map[(x, y, z)] = (True, img.copy(), block_name)
        self._agent_img = (img.copy(), block_name)

    def _clear_belief(self, x: int, y: int, z: int) -> None:
        self._block_map[(x, y, z)] = (False, "air", "air")
        self._agent_img = None
        self._last_targeted_block = None

    def _get_block_from_current_view(self, x: int, y: int, z: int) -> np.ndarray:
        """
        Get the image of the block at the given coordinates from the agent's current view.

        Parameters
        ----------
        x : int
            x-coordinate of the block
        y : int
            y-coordinate of the block
        z : int
            z-coordinate of the block

        Returns
        -------
        np.ndarray
            Image of the block from the current agent's view
        """
        projectionMatrix = self._get_intrinsic_matrix()
        viewMatrix = self._get_extrinsic_matrix()

        # corners of the block
        corners = np.array([
            [x, y, z],
            [x+1, y, z],
            [x+1, y, z+1],
            [x, y, z+1],
            [x, y+1, z],
            [x+1, y+1, z],
            [x+1, y+1, z+1],
            [x, y+1, z+1]
        ])
        # add 1 to the last column for homogeneous coordinates
        corners = np.hstack([corners, np.ones((8, 1))])
        # transform the corners to the agent's view
        corners = corners @ viewMatrix.T
        corners = corners[:, :3]

        # project the corners to the image plane
        pixels = corners @ projectionMatrix.T
        pixels = pixels[:, :2] / pixels[:, 2].reshape(-1, 1)
        pixels = pixels.astype(int)

        # crop the block from the image
        img = self.prev_obs["rgb"].transpose(1, 2, 0)
        x_min = max(pixels[:, 0].min()-10, 0)
        x_max = min(pixels[:, 0].max()+10, img.shape[1])
        y_min = max(pixels[:, 1].min()-10, 0)
        y_max = min(pixels[:, 1].max()+10, img.shape[0])
        img = img[y_min:y_max, x_min:x_max]

        # resize the image to (32, 32)
        img = Image.fromarray(img)
        img = img.resize((32, 32))
        img = np.array(img)
        return img

    def _teleport(self, x: int, y: int, z: int) -> None:
        x_t = x + np.random.uniform(0.3, 0.7)
        y_t = y
        z_t = z + np.random.uniform(0.3, 0.7)
        current_yaw = self.prev_obs["location_stats"]["yaw"][0]
        current_pitch = self.prev_obs["location_stats"]["pitch"][0]
        self._env.teleport_agent(x_t, y_t, z_t, current_yaw, current_pitch)
        self._wait(2)

    def _look_at_block(self, x, y, z, noisy=False) -> None:
        """
        Look at the block at the given coordinates.

        Parameters
        ----------
        x : int
            x-coordinate of the block
        y : int
            y-coordinate of the block
        z : int
            z-coordinate of the block
        noisy : bool, optional
            Add noise to the agent's view, by default False

        Returns
        -------
        None
        """
        # get the center of the block
        ox = x + 0.5
        oy = y + 0.5
        oz = z + 0.5
        agent_x, agent_y, agent_z = self.prev_obs["location_stats"]["pos"]
        # get the agent's eye level
        eye_y = agent_y + 1 + 10/16
        dx = ox - agent_x
        dy = oy - eye_y
        dz = oz - agent_z
        yaw = np.degrees(np.arctan2(-dx, dz))
        pitch = np.degrees(np.arctan2(-dy, np.sqrt(dx**2 + dz**2)))
        if noisy:
            yaw += np.random.normal(0, 1.5)
            pitch += np.random.normal(0, 1.5)
        yaw = np.clip(yaw, -180, 180)
        pitch = np.clip(pitch, -89, 89)
        self._env.teleport_agent(agent_x, agent_y, agent_z, yaw, pitch)
        self._wait(2)

    def _block_exists(self, x, y, z) -> bool:
        if (x, y, z) not in self._block_map:
            return False
        return self._block_map[(x, y, z)][0]

    def _get_intrinsic_matrix(self) -> np.ndarray:
        fov = 70
        width = 800
        height = 600
        f = 1 / np.tan(np.radians(fov) / 2)
        aspect = width / height
        fw = (f/aspect) * (width / 2)
        fh = f * (height / 2)
        intrinsic_matrix = np.array([
            [fw, 0, (width - 1) / 2],
            [0, fh, (height - 1) / 2],
            [0, 0, 1],
        ])
        return intrinsic_matrix

    def _get_extrinsic_matrix(self) -> np.ndarray:
        agent_x, agent_y, agent_z = self.prev_obs["location_stats"]["pos"]
        agent_y = agent_y + 1 + 10/16
        pitch = np.radians(self.prev_obs["location_stats"]["pitch"])[0]
        yaw = np.radians(self.prev_obs["location_stats"]["yaw"])[0]
        yawMatrix = np.array([
            [np.cos(yaw), 0, -np.sin(yaw)],
            [0, 1, 0],
            [np.sin(yaw), 0, np.cos(yaw)],
        ])
        pitchMatrix = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ])
        rotationMatrix = yawMatrix @ pitchMatrix
        agent_pos = np.array([agent_x, agent_y, agent_z]).reshape(-1, 1)
        world_to_agent = np.hstack([rotationMatrix.T, -rotationMatrix.T @ agent_pos])
        world_to_agent = np.vstack([world_to_agent, np.array([0, 0, 0, 1])])
        return world_to_agent

    @staticmethod
    def NO_OP() -> list:
        return [0, 0, 0, 12, 12, 0, 0, 0]

    @staticmethod
    def ATTACK() -> list:
        return [0, 0, 0, 12, 12, 3, 0, 0]

    @staticmethod
    def CRAFT(item_id: int) -> list:
        return [0, 0, 0, 12, 12, 4, item_id, 0]

    @staticmethod
    def EQUIP(inv_slot_idx: int):
        return [0, 0, 0, 12, 12, 5, 0, inv_slot_idx]

    @staticmethod
    def PLACE(inv_slot_idx: int):
        return [0, 0, 0, 12, 12, 6, 0, inv_slot_idx]

    @staticmethod
    def blocks_to_xml(blocks):
        """
        Original source: https://github.com/IretonLiu/mine-pddl
        Converts a list of blocks to an xml string
        """
        xml_str = ""
        for b in blocks:
            xml_str += f"""<DrawBlock x=\"{b['position']['x']}\""""
            xml_str += f""" y=\"{b['position']['y']}\""""
            xml_str += f""" z=\"{b['position']['z']}\""""
            xml_str += f""" type=\"{b['type']}\"/>"""
        return xml_str

    @staticmethod
    def inventory_to_items(inventory):
        """
        Original source: https://github.com/IretonLiu/mine-pddl
        Converts a list of inventory items to an inventory item list
        """
        inventory_item_list = []
        for i, item in enumerate(inventory):
            if item["type"] == "air" or item["quantity"] == 0:
                continue
            inventory_item_list.append(
                minedojo.sim.InventoryItem(
                    slot=i,
                    name=item["type"],
                    variant=item["variant"] if "variant" in item else None,
                    quantity=item["quantity"],
                )
            )
        return None if len(inventory_item_list) == 0 else inventory_item_list


class MinecraftDataset(UnorderedDataset):
    ACTION_TO_IDX = {
        "teleport": 0,
        "attack": 1,
        "place": 2,
        "craft": 3
    }
    DIRECTION_TO_IDX = {
        "north": 4,
        "south": 5,
        "east": 6,
        "west": 7,
    }
    ITEMS_TO_IDX = {x: i for i, x in enumerate(USEFUL_ITEMS, 8)}

    def __getitem__(self, idx):
        x, x_, key_order = dict_to_transition(self._state[idx], self._next_state[idx],
                                              exclude_keys=self.exclude_keys)
        if not self._privileged:
            x = self._normalize_imgs(x)
            x_ = self._normalize_imgs(x_)
        if self._transform_action:
            a = self._actions_to_label(self._action[idx], key_order)
        else:
            a = self._action[idx]
        return x, a, x_

    def _normalize_imgs(self, x):
        x["agent"] = x["agent"] / 255.0
        x["inventory"] = x["inventory"]
        x["objects"][..., :3072] = x["objects"][..., :3072] / 255.0
        return x

    @staticmethod
    def _actions_to_label(action, key_order):
        n_action = len(USEFUL_ITEMS) + 8
        action_type, action_args, object_args = action
        a_ = torch.zeros(n_action, dtype=torch.float32)
        a_[MinecraftDataset.ACTION_TO_IDX[action_type]] = 1
        if (action_type == "teleport"):
            a_[MinecraftDataset.DIRECTION_TO_IDX[action_args[0]]] = 1
        else:
            a_[MinecraftDataset.ITEMS_TO_IDX[action_args[0]]] = 1

        if len(object_args) != 0:
            target_obj = object_args[0]
            target_obj = key_order.index(target_obj) + 1
        else:
            target_obj = 0
        a = torch.zeros(len(key_order)+1, n_action, dtype=torch.float32)
        a[target_obj] = a_
        return a

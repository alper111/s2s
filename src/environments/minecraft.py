from PIL import Image
from typing import Optional
import pickle
import os

import minedojo
from minedojo.sim.mc_meta.mc import ALL_ITEMS, ALL_CRAFT_SMELT_ITEMS
import gym
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from s2s.helpers import dict_to_transition

USEFUL_ITEMS = {
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
}


class Minecraft(gym.Env):
    def __init__(self, world_config: dict, world_seed: int = 0, seed: int = 0):
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
        self._prev_obs = None

    @property
    def observation(self) -> dict:
        obs = {}
        if self._agent_img is not None:
            img = self._agent_img.copy().reshape(-1) / 255.0
            ax, ay, az = self.agent_pos
            east_exists = self._block_exists(ax+1, ay, az)
            south_exists = self._block_exists(ax, ay, az+1)
            west_exists = self._block_exists(ax-1, ay, az)
            north_exists = self._block_exists(ax, ay, az-1)
            top_exists = self._block_exists(ax, ay+1, az)
            neighbors = np.array([east_exists, south_exists, west_exists, north_exists, top_exists], dtype=float)
            obs["agent"] = {0: np.concatenate([img, neighbors])}
        else:
            obs["agent"] = {0: None}
        inventory_img = np.transpose(self.prev_obs["rgb"][:, 558:598, 220:580], (1, 2, 0))
        inventory_img = Image.fromarray(inventory_img)
        inventory_img = inventory_img.resize((32*9, 32))
        inventory_img = np.array(inventory_img).reshape(-1) / 255.0
        obs["inventory"] = {1: inventory_img}
        obs["objects"] = {}
        for key in self._block_map:
            if self._block_map[key][0]:
                x, y, z = key
                img = self._block_map[key][1].copy().reshape(-1) / 255.0
                east_exists = self._block_exists(x+1, y, z)
                south_exists = self._block_exists(x, y, z+1)
                west_exists = self._block_exists(x-1, y, z)
                north_exists = self._block_exists(x, y, z-1)
                top_exists = self._block_exists(x, y+1, z)
                neighbors = np.array([east_exists, south_exists, west_exists, north_exists, top_exists], dtype=float)
                obs["objects"][key] = np.concatenate([img, neighbors])
        obs["dimensions"] = {"inventory": 32*32*3*9, "agent": 32*32*3+5, "objects": 32*32*3+5}
        return obs

    @property
    def reward(self) -> float:
        return 0

    @property
    def done(self) -> bool:
        available_actions = self.available_actions()
        return len(available_actions) == 0

    @property
    def info(self) -> dict:
        return self.prev_obs

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
    def last_targeted_block(self) -> tuple[int, int, int]:
        return self._last_targeted_block

    @property
    def traced_block(self) -> tuple[int, int, int]:
        x = int(np.floor(self.prev_obs["rays"]["traced_block_x"]))
        y = int(np.floor(self.prev_obs["rays"]["traced_block_y"]))
        z = int(np.floor(self.prev_obs["rays"]["traced_block_z"]))
        return x, y, z

    def reset(self) -> dict:
        obs = self._env.reset()
        self._prev_obs = obs
        self._observe_all_blocks()
        return self.observation

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        action_type, (action_args, object_args) = action
        if action_type == "teleport":
            x, y, z = object_args[0]
            side = action_args[0]
            res = self._teleport_to_block(x, y, z, side)
        elif action_type == "attack":
            res = self._attack_block()
        elif action_type == "place":
            block_name = action_args[0]
            res = self._place_block(block_name)
        elif action_type == "equip":
            item_name = action_args[0]
            res = self._equip_item(item_name)
        elif action_type == "craft":
            item_name = action_args[0]
            res = self._craft_item(item_name)
        else:
            raise ValueError("Invalid action type")

        obs = self.observation
        reward = self.reward
        done = self.done
        info = self.info
        info["action_success"] = res
        return obs, reward, done, info

    def sample_action(self):
        actions = self.available_actions()
        action_types = list(actions.keys())
        available_action_types = [a for a in action_types if len(actions[a]) > 0]
        a = np.random.choice(available_action_types)
        args = actions[a][np.random.randint(0, len(actions[a]))]
        return (a, args)

    def available_actions(self):
        actions = {"teleport": [],
                   "attack": [],
                   "place": [],
                   "equip": [],
                   "craft": []}
        # possible teleport locations
        # teleport(block, side)
        for (x, y, z) in self._block_map:
            if not self._block_map[(x, y, z)][0]:
                continue
            if (not self._block_map[(x+1, y, z)][0]) and self._below_support_exists(x+1, y, z):
                actions["teleport"].append((("east",), ((x, y, z),)))
            if (not self._block_map[(x, y, z+1)][0]) and self._below_support_exists(x, y, z+1):
                actions["teleport"].append((("south",), ((x, y, z),)))
            if (not self._block_map[(x-1, y, z)][0]) and self._below_support_exists(x-1, y, z):
                actions["teleport"].append((("west",), ((x, y, z),)))
            if (not self._block_map[(x, y, z-1)][0]) and self._below_support_exists(x, y, z-1):
                actions["teleport"].append((("north",), ((x, y, z),)))
            if (not self._block_map[(x, y+1, z)][0]) and self._below_support_exists(x, y+1, z):
                actions["teleport"].append((("top",), ((x, y, z),)))

        if self.last_targeted_block is not None:
            # attack()
            actions["attack"].append(((), ()))
        for item_type in self._placeable_items():
            # place(item_type)
            actions["place"].append(((item_type,), ()))
        for item_type in self._equippable_items():
            # equip(item_type)
            actions["equip"].append(((item_type,), ()))
        for item_type in self._craftable_items():
            if item_type not in USEFUL_ITEMS:
                continue
            # craft(craftable_item)
            actions["craft"].append(((item_type,), ()))
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
        elif side == "top":
            self._teleport(x, y+1, z)
        self._look_and_update_belief(x, y, z)
        return True

    def _attack_block(self) -> bool:
        x, y, z = self.last_targeted_block
        self._look_at_block(x, y, z)
        if self.traced_block != (x, y, z):
            return False
        exists = self._block_exists(x, y, z)
        if exists:
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
        jump = Minecraft.JUMP_AND_LOOK_DOWN()
        self._step(jump)

        place_idx = Minecraft.PLACE(inventory_slot)
        self._step(place_idx)
        self._look_and_update_belief(ax, ay, az)
        if (self._world_limit["xmin"] < ax) and ((ax-1, ay, az) not in self._block_map):
            self._block_map[(ax-1, ay, az)] = (False, "air")
        if (self._world_limit["xmax"] > ax) and ((ax+1, ay, az) not in self._block_map):
            self._block_map[(ax+1, ay, az)] = (False, "air")
        if (self._world_limit["ymin"] < ay) and ((ax, ay-1, az) not in self._block_map):
            self._block_map[(ax, ay-1, az)] = (False, "air")
        if (self._world_limit["ymax"] > ay) and ((ax, ay+1, az) not in self._block_map):
            self._block_map[(ax, ay+1, az)] = (False, "air")
        if (self._world_limit["zmin"] < az) and ((ax, ay, az-1) not in self._block_map):
            self._block_map[(ax, ay, az-1)] = (False, "air")
        if (self._world_limit["zmax"] > az) and ((ax, ay, az+1) not in self._block_map):
            self._block_map[(ax, ay, az+1)] = (False, "air")
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

    def _get_item_index(self, item_name: str) -> Optional[int]:
        indices, = np.where(self.prev_obs["inventory"]["name"] == item_name.replace("_", " "))
        if len(indices) == 0:
            return None
        return indices[0]

    def _equippable_items(self):
        equip_mask = self.prev_obs["masks"]["equip"]
        item_names = np.unique(self.prev_obs["inventory"]["name"][equip_mask])
        item_names = [name.replace(" ", "_") for name in item_names]
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
                self._block_map[(x_i-1, y_i, z_i)] = (False, "air")
            if self._world_limit["xmax"] > x_i:
                self._block_map[(x_i+1, y_i, z_i)] = (False, "air")
            if self._world_limit["ymin"] < y_i:
                self._block_map[(x_i, y_i-1, z_i)] = (False, "air")
            if self._world_limit["ymax"] > y_i:
                self._block_map[(x_i, y_i+1, z_i)] = (False, "air")
            if self._world_limit["zmin"] < z_i:
                self._block_map[(x_i, y_i, z_i-1)] = (False, "air")
            if self._world_limit["zmax"] > z_i:
                self._block_map[(x_i, y_i, z_i+1)] = (False, "air")

    def _look_and_update_belief(self, x: int, y: int, z: int) -> None:
        self._look_at_block(x, y, z, noisy=True)
        img = self._get_block_from_current_view(x, y, z)
        self._last_targeted_block = (x, y, z)
        self._block_map[(x, y, z)] = (True, img.copy())
        self._agent_img = img.copy()

    def _clear_belief(self, x: int, y: int, z: int) -> None:
        self._block_map[(x, y, z)] = (False, "air")
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
        self._env.teleport_agent(agent_x, agent_y, agent_z, yaw, pitch)
        self._wait(2)

    def _block_exists(self, x, y, z) -> bool:
        if (x, y, z) not in self._block_map:
            return False
        ax, ay, az = self.agent_pos
        if (x == ax) and (y == ay) and (z == az):
            return True
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
    def JUMP_AND_LOOK_DOWN():
        return [0, 0, 1, 24, 12, 0, 0, 0]

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


class MinecraftAbstractionDataset(torch.utils.data.Dataset):
    ACTION_TO_IDX = {
        "teleport": 0,
        "attack": 1,
        "place": 2,
        "equip": 3,
        "craft": 4
    }
    DIRECTION_TO_IDX = {
        "north": 5,
        "south": 6,
        "east": 7,
        "west": 8,
        "top": 9
    }
    ITEMS_TO_IDX = {x: i for i, x in enumerate(ALL_ITEMS, 10)}

    def __init__(self, root_folder):
        self._root_folder = root_folder
        self._state = np.load(os.path.join(root_folder, "state.npy"), allow_pickle=True)
        self._action = pickle.load(open("out/action.pkl", "rb"))
        self._next_state = np.load(os.path.join(root_folder, "next_state.npy"), allow_pickle=True)

    def __len__(self):
        return len(self._state)

    def __getitem__(self, idx):
        x, x_, keys = dict_to_transition(self._state[idx], self._next_state[idx])
        a_ = self._actions_to_label(self._action[idx])
        target = self._action[idx][1][1]
        if len(target) != 0:
            target = target[0]
            target = keys.index(target)
        else:
            target = 0
        a = torch.zeros(len(keys), a_.shape[0], dtype=torch.float32)
        a[target] = a_
        return x, a, x_

    @staticmethod
    def _actions_to_label(action):
        action_type, args = action
        action_args, _ = args
        a = torch.zeros(402, dtype=torch.float32)
        a[MinecraftAbstractionDataset.ACTION_TO_IDX[action_type]] = 1
        if action_type == "teleport":
            a[MinecraftAbstractionDataset.DIRECTION_TO_IDX[action_args[0]]] = 1
        elif action_type != "attack":
            a[MinecraftAbstractionDataset.ITEMS_TO_IDX[action_args[0]]] = 1
        return a

    @staticmethod
    def collate_fn(batch):
        x, a, x_ = batch[0]
        keys = list(x.keys())
        s = {k: [] for k in keys}
        s_ = {k: [] for k in keys}
        s["masks"] = {k: [] for k in keys}
        s_["masks"] = {k: [] for k in keys}
        a = []
        for x, a_, x_ in batch:
            for k in keys:
                s[k].append(x[k])
                s_[k].append(x_[k])
                s["masks"][k].append(torch.ones(x[k].shape[0], dtype=torch.bool))
                s_["masks"][k].append(torch.ones(x_[k].shape[0], dtype=torch.bool))
            a.append(a_)
        for k in keys:
            s[k] = pad_sequence(s[k], batch_first=True)
            s_[k] = pad_sequence(s_[k], batch_first=True)
            s["masks"][k] = pad_sequence(s["masks"][k], batch_first=True)
            s_["masks"][k] = pad_sequence(s_["masks"][k], batch_first=True)
        a = pad_sequence(a, batch_first=True)
        return s, a, s_

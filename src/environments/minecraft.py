from PIL import Image

import minedojo
import gym
import numpy as np


class Minecraft(gym.Env):

    NO_OP = 0
    USE = 1
    DROP = 2
    ATTACK = 3
    CRAFT = 4
    EQUIP = 5
    PLACE = 6
    DESTROY = 7

    def __init__(self, world_config: dict, world_seed: int = 0, seed: int = 0):
        drawing_str = None
        initial_inventory = None
        self._max_objects = 0
        voxel_range = {"xmin": -1, "xmax": 1, "ymin": -1, "ymax": 1, "zmin": -1, "zmax": 1}
        if "blocks" in world_config:
            drawing_str = Minecraft.blocks_to_xml(world_config["blocks"])
            self._max_objects = len(world_config["blocks"]) + 1
        if "inventory" in world_config:
            initial_inventory = Minecraft.inventory_to_items(world_config["inventory"])
        if "voxel_range" in world_config:
            voxel_range = world_config["voxel_range"]
        self.voxel_range = voxel_range

        self._env = minedojo.make(
            task_id="open-ended",
            generate_world_type="flat",
            image_size=(600, 800),
            world_seed=world_seed,
            seed=seed,
            use_voxel=True,
            voxel_size=self.voxel_range,
            drawing_str=drawing_str,
            initial_inventory=initial_inventory,
            start_position=dict(x=0.5, y=4, z=0.5, yaw=0, pitch=0),
            break_speed_multiplier=1000,
            allow_mob_spawn=False,
            use_lidar=True,
            lidar_rays=[(0, 0, 100)],
            allow_time_passage=False
        )

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self._max_objects, 32, 32, 3))
        self.action_space = gym.spaces.Box(low=-255, high=255, shape=(4,))

        self._prev_obs = {}
        self._order = []

    @property
    def observation(self) -> np.ndarray:
        if self._order == []:
            self._order = list(self._prev_obs.keys())
        return np.array([self._prev_obs[key] for key in self._order])

    @property
    def reward(self) -> float:
        return 0

    @property
    def done(self) -> bool:
        available_actions = self.available_actions()
        return len(available_actions) == 0

    @property
    def info(self) -> dict:
        return self._env.prev_obs

    @property
    def object_centric(self) -> bool:
        return True

    @property
    def agent_pos(self) -> tuple[int, int, int]:
        pos = self._env.prev_obs['location_stats']['pos']
        return int(np.floor(pos[0])), int(np.floor(pos[1])), int(np.floor(pos[2]))

    def reset(self) -> np.ndarray:
        self._env.reset()
        self._observe_all_in_voxel_range()
        return self.observation

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        action_type = action[0]
        print(action)
        if action_type == 0:
            x, y, z = action[1], action[2], action[3]
            self.teleport_to_block(x, y, z)
        elif action_type == 1:
            x, y, z = action[1], action[2], action[3]
            self.attack_block(x, y, z)

        obs = self.observation
        reward = self.reward
        done = self.done
        info = self.info
        return obs, reward, done, info

    def sample_action(self) -> tuple:
        available_actions = self.available_actions()
        r = np.random.randint(0, len(available_actions))
        return available_actions[r]

    def available_actions(self):
        blocks = self._get_blocks_in_range()
        actions = []
        for (x, y, z) in blocks:
            actions.append((0, x, y, z))
            actions.append((1, x, y, z))
        return actions

    def teleport_to_block(self, x: int, y: int, z: int):
        agent_y = self.agent_pos[1]
        if agent_y == y:
            self._look_at_block(x, y, z, teleport_nearby=True, noisy=True)
            img = self._observe_block(x, y, z)
            self._prev_obs['agent'] = img.copy()
            self._prev_obs[(x, y, z)] = img.copy()
        return img

    def attack_block(self, x: int, y: int, z: int):
        self._look_at_block(x, y, z)
        exists = self._block_exists(x, y, z)
        if exists:
            action = self._env.action_space.no_op()
            action[5] = Minecraft.ATTACK
            self._env.step(action)
            for _ in range(2):
                self._env.step(self._env.action_space.no_op())
            img = self._observe_block(x, y, z)
        else:
            self._env.step(self._env.action_space.no_op())
            img = np.zeros((32, 32, 3), dtype=np.uint8)
        self._prev_obs[(x, y, z)] = img.copy()
        self._prev_obs['agent'] = img.copy()
        return img

    def close(self):
        self._env.close()

    def _observe_all_in_voxel_range(self):
        blocks = self._get_blocks_in_range()
        for (x_i, y_i, z_i) in blocks:
            self._look_at_block(x_i, y_i, z_i, teleport_nearby=True, noisy=True)
            pixels = self._observe_block(x_i, y_i, z_i)
            self._prev_obs[(x_i, y_i, z_i)] = pixels.copy()
        self._prev_obs['agent'] = pixels.copy()
        return self.observation

    def _get_blocks_in_range(self):
        ox, oy, oz = np.where(self._env.prev_obs['voxels']['is_solid'])
        x, y, z = self.agent_pos
        blocks = []
        for x_i, y_i, z_i in zip(ox, oy, oz):
            x_t = x + x_i + self.voxel_range['xmin']
            y_t = y + y_i + self.voxel_range['ymin']
            z_t = z + z_i + self.voxel_range['zmin']
            blocks.append((x_t, y_t, z_t))
        return blocks

    def _observe_block(self, x: int, y: int, z: int) -> np.ndarray:
        """
        Observe and crop the block at the given coordinates.

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
            Cropped image of the block with (32, 32, 3) shaped array.
        """
        if not self._block_exists(x, y, z):
            return np.zeros((32, 32, 3), dtype=np.uint8)

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
        img = self._env.prev_obs['rgb'].transpose(1, 2, 0)
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

    def _look_at_block(self, x, y, z, teleport_nearby=False, noisy=False) -> None:
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
        teleport_nearby : bool, optional
            Teleport the agent to a random location near the block, by default False
        noisy : bool, optional
            Add noise to the agent's view, by default False

        Returns
        -------
        None
        """
        # get the center of the block
        ox, oy, oz = x+0.5, y+0.5, z+0.5
        agent_x, agent_y, agent_z = self._env.prev_obs['location_stats']['pos']
        if teleport_nearby:
            radius = np.random.uniform(2, 2.5)  # teleport within 1 to 1.5 blocks
            theta = np.radians(np.random.uniform(0, 360))
            x_t = ox + radius * np.cos(theta)
            y_t = oy - 0.5
            z_t = oz + radius * np.sin(theta)
        else:
            x_t, y_t, z_t = agent_x, agent_y, agent_z

        eye_y = y_t + 1 + 10/16
        dx, dy, dz = ox - x_t, oy - eye_y, oz - z_t

        yaw = np.degrees(np.arctan2(-dx, dz))
        pitch = np.degrees(np.arctan2(-dy, np.sqrt(dx**2 + dz**2)))
        if noisy:
            yaw += np.random.normal(0, 1.5)
            pitch += np.random.normal(0, 1.5)

        self._env.teleport_agent(x_t, y_t, z_t, yaw, pitch)
        for _ in range(2):
            self._env.step(self._env.action_space.no_op())

    def _block_exists(self, x, y, z) -> bool:
        ax, ay, az = self.agent_pos
        dx, dy, dz = x - ax, y - ay, z - az
        dx = dx - self.voxel_range['xmin']
        dy = dy - self.voxel_range['ymin']
        dz = dz - self.voxel_range['zmin']
        if dx < 0 or dx > (self.voxel_range['xmax'] - self.voxel_range['xmin']):
            return False
        if dy < 0 or dy > (self.voxel_range['ymax'] - self.voxel_range['ymin']):
            return False
        if dz < 0 or dz > (self.voxel_range['zmax'] - self.voxel_range['zmin']):
            return False
        if not self._env.prev_obs['voxels']['is_solid'][dx, dy, dz]:
            return False
        return True

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
        agent_x, agent_y, agent_z = self._env.prev_obs['location_stats']['pos']
        agent_y = agent_y + 1 + 10/16
        pitch = np.radians(self._env.prev_obs['location_stats']['pitch'])[0]
        yaw = np.radians(self._env.prev_obs['location_stats']['yaw'])[0]
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

    @staticmethod
    def get_delta_mask(state: np.ndarray, next_state: np.ndarray) -> np.ndarray:
        return state != next_state

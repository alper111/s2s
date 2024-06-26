import minedojo
import numpy as np


def look_at_block(env, x, y, z, teleport_nearby=False, noisy=False):
    # get the center of the block
    ox, oy, oz = x+0.5, y+0.5, z+0.5
    agent_x, agent_y, agent_z = env.prev_obs['location_stats']['pos']
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
        yaw += np.random.normal(0, 5)
        pitch += np.random.normal(0, 5)

    env.teleport_agent(x_t, y_t, z_t, yaw, pitch)
    for _ in range(5):
        obs, _, _, _ = env.step(env.action_space.no_op())
    return obs


def observe(env, x, y, z):
    agent_x, agent_y, agent_z = env.prev_obs['location_stats']['pos']
    # add eye-height to y, which is 1 + 10/16
    agent_y = agent_y + 1 + 10/16
    pitch = np.radians(env.prev_obs['location_stats']['pitch'])[0]
    yaw = np.radians(env.prev_obs['location_stats']['yaw'])[0]
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
    corners = np.hstack([corners, np.ones((8, 1))])
    corners = corners @ world_to_agent.T
    corners = corners[:, :3]

    fov = 70
    width = 800
    height = 600
    f = 1 / np.tan(np.radians(fov) / 2)
    aspect = width / height
    fw = (f/aspect) * (width / 2)
    fh = f * (height / 2)
    projectionMatrix = np.array([
        [-fw, 0, (width - 1) / 2],
        [0, -fh, (height - 1) / 2],
        [0, 0, 1],
    ])
    pixel_points = corners @ projectionMatrix.T
    pixel_points = pixel_points[:, :2] / pixel_points[:, 2].reshape(-1, 1)
    pixel_points = pixel_points.astype(int)
    return pixel_points


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

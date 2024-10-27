from ursina import *
import time

class Game(Ursina.__closure__[0].cell_contents):
    def __init__(self):
        super().__init__()
        window.fullscreen = False
        Entity(model='sphere', scale=100, texture='textures/Solid_black.png', double_sided=True)
        EditorCamera()
        camera.world_position = (0, 0, -15)
        self.model, self.texture = 'models/custom_cube', 'textures/rubik_texture'
        self.load_game()

    def load_game(self):
        self.create_cube_positions()
        self.CUBES = [Entity(model=self.model, texture=self.texture, position=pos) for pos in self.SIDE_POSITIONS]
        self.PARENT = Entity()
        self.rotation_axes = {'LEFT': 'x', 'RIGHT': 'x', 'TOP': 'y', 'BOTTOM': 'y', 'FACE': 'z', 'BACK': 'z'}
        self.cubes_side_positons = {'LEFT': self.LEFT, 'BOTTOM': self.BOTTOM, 'RIGHT': self.RIGHT, 'FACE': self.FACE,
                                    'BACK': self.BACK, 'TOP': self.TOP}
        self.animation_time = 0.5
        self.action_trigger = True

    def perform_automated_moves(self):
        '''Automate a sequence of cube moves'''
        invoke(self.rotate_side, 'RIGHT', delay=1)
        invoke(self.rotate_side, 'TOP', delay=2)
        invoke(self.rotate_side_prime, 'RIGHT', delay=3)
        invoke(self.rotate_side_prime, 'TOP', delay=4)
        invoke(self.rotate_side, 'RIGHT', delay=5)
        invoke(self.rotate_side, 'TOP', delay=6)
        invoke(self.rotate_side_prime, 'RIGHT', delay=7)
        invoke(self.rotate_side_prime, 'TOP', delay=8)
        invoke(self.rotate_side, 'RIGHT', delay=9)
        invoke(self.rotate_side, 'TOP', delay=10)
        invoke(self.rotate_side_prime, 'RIGHT', delay=11)
        invoke(self.rotate_side_prime, 'TOP', delay=12)
        invoke(self.rotate_side, 'RIGHT', delay=13)
        invoke(self.rotate_side, 'TOP', delay=14)
        invoke(self.rotate_side_prime, 'RIGHT', delay=15)
        invoke(self.rotate_side_prime, 'TOP', delay=16)
        invoke(self.rotate_side, 'RIGHT', delay=17)
        invoke(self.rotate_side, 'TOP', delay=18)
        invoke(self.rotate_side_prime, 'RIGHT', delay=19)
        invoke(self.rotate_side_prime, 'TOP', delay=20)
        invoke(self.rotate_side, 'RIGHT', delay=21)
        invoke(self.rotate_side, 'TOP', delay=22)
        invoke(self.rotate_side_prime, 'RIGHT', delay=23)
        invoke(self.rotate_side_prime, 'TOP', delay=24)

    def toggle_animation_trigger(self):
        self.action_trigger = not self.action_trigger

    def rotate_side(self, side_name):
        self.action_trigger = False
        cube_positions = self.cubes_side_positons[side_name]
        rotation_axis = self.rotation_axes[side_name]
        self.reparent_to_scene()
        for cube in self.CUBES:
            if cube.position in cube_positions:
                cube.parent = self.PARENT
                eval(f'self.PARENT.animate_rotation_{rotation_axis}(90, duration=self.animation_time)')
        invoke(self.toggle_animation_trigger, delay=self.animation_time + 0.11)

    def rotate_side_prime(self, side_name):
        self.action_trigger = False
        cube_positions = self.cubes_side_positons[side_name]
        rotation_axis = self.rotation_axes[side_name]
        self.reparent_to_scene()
        for cube in self.CUBES:
            if cube.position in cube_positions:
                cube.parent = self.PARENT
                eval(f'self.PARENT.animate_rotation_{rotation_axis}(-90, duration=self.animation_time)')
        invoke(self.toggle_animation_trigger, delay=self.animation_time + 0.11)

    def reparent_to_scene(self):
        for cube in self.CUBES:
            if cube.parent == self.PARENT:
                world_pos, world_rot = round(cube.world_position, 1), cube.world_rotation
                cube.parent = scene
                cube.position, cube.rotation = world_pos, world_rot
        self.PARENT.rotation = 0

    def create_cube_positions(self):
        self.LEFT = {Vec3(-1, y, z) for y in range(-1, 2) for z in range(-1, 2)}
        self.BOTTOM = {Vec3(x, -1, z) for x in range(-1, 2) for z in range(-1, 2)}
        self.FACE = {Vec3(x, y, -1) for x in range(-1, 2) for y in range(-1, 2)}
        self.BACK = {Vec3(x, y, 1) for x in range(-1, 2) for y in range(-1, 2)}
        self.RIGHT = {Vec3(1, y, z) for y in range(-1, 2) for z in range(-1, 2)}
        self.TOP = {Vec3(x, 1, z) for x in range(-1, 2) for z in range(-1, 2)}
        self.SIDE_POSITIONS = self.LEFT | self.BOTTOM | self.FACE | self.BACK | self.RIGHT | self.TOP

    def input(self, key, event=None):
        keys = dict(zip('asdwqe', 'LEFT BOTTOM RIGHT TOP FACE BACK'.split()))
        keys_p = dict(zip('jkliuo', 'LEFT BOTTOM RIGHT TOP FACE BACK'.split()))
        if self.action_trigger:
            if key in keys:
                self.rotate_side(keys[key])
            if key in keys_p:
                self.rotate_side_prime(keys_p[key])
        super().input(key)


if __name__ == '__main__':
    game = Game()
    game.run()
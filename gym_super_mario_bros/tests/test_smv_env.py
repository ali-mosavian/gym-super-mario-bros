"""Test cases for the Super Mario Bros meta environment."""
from unittest import TestCase
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv


class ShouldRaiseErrorOnInvalidRomMode(TestCase):
    def test(self):
        self.assertRaises(ValueError, SuperMarioBrosEnv, rom_mode=-1)
        self.assertRaises(ValueError, SuperMarioBrosEnv, rom_mode=5)
        self.assertRaises(ValueError, SuperMarioBrosEnv, rom_mode=-1, lost_levels=True)
        self.assertRaises(ValueError, SuperMarioBrosEnv, rom_mode=5, lost_levels=True)


class ShouldRaiseErrorOnInvalidTypeLostLevels(TestCase):
    def test(self):
        self.assertRaises(TypeError, SuperMarioBrosEnv, lost_levels='foo')


class ShouldRaiseErrorOnInvalidTypeWorld(TestCase):
    def test(self):
        self.assertRaises(TypeError, SuperMarioBrosEnv, target=('foo', 1))


class ShouldRaiseErrorOnBelowBoundsWorld(TestCase):
    def test(self):
        self.assertRaises(ValueError, SuperMarioBrosEnv, target=(0, 1))
        self.assertRaises(ValueError, SuperMarioBrosEnv, target=(0, 1), lost_levels=True)


class ShouldRaiseErrorOnAboveBoundsWorld(TestCase):
    def test(self):
        self.assertRaises(ValueError, SuperMarioBrosEnv, target=(9, 1))
        self.assertRaises(ValueError, SuperMarioBrosEnv, target=(13, 1), lost_levels=True)


class ShouldRaiseErrorOnInvalidTypeStage(TestCase):
    def test(self):
        self.assertRaises(TypeError, SuperMarioBrosEnv, target=('foo', 1))


class ShouldRaiseErrorOnBelowBoundsStage(TestCase):
    def test(self):
        self.assertRaises(ValueError, SuperMarioBrosEnv, target=(1, 0))
        self.assertRaises(ValueError, SuperMarioBrosEnv, target=(1, 0), lost_levels=True)


class ShouldRaiseErrorOnAboveBoundsStage(TestCase):
    def test(self):
        self.assertRaises(ValueError, SuperMarioBrosEnv, target=(1, 5))
        self.assertRaises(ValueError, SuperMarioBrosEnv, target=(1, 5), lost_levels=True)


class ShouldStepGameEnv(TestCase):
    def test(self):
        env = SuperMarioBrosEnv()
        self.assertFalse(env.unwrapped.is_single_stage_env)
        self.assertIsNone(env.unwrapped._target_world)
        self.assertIsNone(env.unwrapped._target_stage)
        self.assertIsNone(env.unwrapped._target_area)
        env.reset()
        state, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(0, info['coins'])
        self.assertEqual(False, info['flag_get'])
        self.assertEqual(2, info['life'])
        self.assertEqual(1, info['world'])
        self.assertEqual(0, info['score'])
        self.assertEqual(1, info['stage'])
        self.assertEqual(400, info['time'])
        self.assertEqual(40, info['x_pos'])
        env.close()


class ShouldStepStageEnv(TestCase):
    def test(self):
        env = SuperMarioBrosEnv(target=(4, 2))
        self.assertTrue(env.unwrapped.is_single_stage_env)
        self.assertIsInstance(env.unwrapped._target_world, int)
        self.assertIsInstance(env.unwrapped._target_stage, int)
        self.assertIsInstance(env.unwrapped._target_area, int)
        env.reset()
        state, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(0, info['coins'])
        self.assertEqual(False, info['flag_get'])
        self.assertEqual(2, info['life'])
        self.assertEqual(4, info['world'])
        self.assertEqual(0, info['score'])
        self.assertEqual(2, info['stage'])
        self.assertEqual(400, info['time'])
        self.assertEqual(40, info['x_pos'])
        env.close()

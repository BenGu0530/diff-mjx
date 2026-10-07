"""Straight-through box-box: hard forward values, finite soft gradients."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
from mujoco.mjx._src import collision_convex
from mujoco.mjx._src import math
from mujoco.mjx._src import soft_collision_test
import numpy as np

_box = soft_collision_test._box


def _rot(axis, angle):
  axis = np.asarray(axis, dtype=np.float32)
  axis = axis / np.linalg.norm(axis)
  k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]])
  return jp.asarray(np.eye(3) + np.sin(angle) * k
                    + (1 - np.cos(angle)) * (k @ k), dtype=jp.float32)


def _random_pairs(n, seed=0):
  """Overlapping, touching and separated box pairs in random poses."""
  rng = np.random.default_rng(seed)
  pairs = []
  for _ in range(n):
    size_a = rng.uniform(0.02, 0.2, 3)
    size_b = rng.uniform(0.02, 0.2, 3)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    gap = rng.uniform(-0.3, 0.1) * min(size_a.min(), size_b.min())
    offset = direction * (np.linalg.norm(size_a) * 0.6 + np.linalg.norm(size_b) * 0.6 + gap)
    box_a = _box(rng.normal(size=3) * 0.01, size=size_a,
                 mat=_rot(rng.normal(size=3), rng.uniform(0, np.pi)))
    box_b = _box(np.asarray(box_a.pos) + offset, size=size_b,
                 mat=_rot(rng.normal(size=3), rng.uniform(0, np.pi)))
    pairs.append((box_a, box_b))
  return pairs


class BoxBoxStraightThroughTest(parameterized.TestCase):

  @parameterized.parameters('smooth', 'c2')
  def test_forward_equals_hard(self, mode):
    for box_a, box_b in _random_pairs(64):
      dist_h, pos_h, n_h = collision_convex._box_box(box_a, box_b)
      frame_h = jax.vmap(math.make_frame)(n_h)
      dist, pos, frame = collision_convex._box_box_straight_through(
          box_a, box_b, mode)
      np.testing.assert_array_equal(np.asarray(dist), np.asarray(dist_h))
      valid = np.asarray(dist_h) < np.finfo(np.float32).max
      np.testing.assert_allclose(np.asarray(pos)[valid],
                                 np.asarray(pos_h)[valid], atol=1e-6)
      np.testing.assert_allclose(np.asarray(frame)[valid],
                                 np.asarray(frame_h)[valid], atol=1e-6)

  @parameterized.parameters('smooth', 'c2')
  def test_gradients_finite_and_nonzero(self, mode):
    nonzero = 0
    for box_a, box_b in _random_pairs(16, seed=1):

      def loss(p, box_a=box_a, box_b=box_b):
        dist, pos, frame = collision_convex._box_box_straight_through(
            box_a.replace(pos=p), box_b, mode)
        valid = dist < jp.finfo(dist.dtype).max
        return jp.sum(jp.where(valid, dist + pos.sum(-1) + frame[:, 0].sum(-1), 0.0))

      g = np.asarray(jax.grad(loss)(box_a.pos))
      self.assertTrue(np.isfinite(g).all())
      nonzero += int(np.abs(g).sum() > 0)
    self.assertGreater(nonzero, 0)


if __name__ == '__main__':
  absltest.main()

"""Hard forward values and surrogate gradients for softened contact helpers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
from mujoco.mjx._src import collision_convex
from mujoco.mjx._src import math
from mujoco.mjx._src import soft_collision_test
from mujoco.mjx._src import softjax as sj
import numpy as np


class ContactStraightThroughTest(parameterized.TestCase):

  @parameterized.product(
      operation=('abs', 'sign', 'relu', 'clip', 'less', 'greater',
                 'greater_equal', 'max', 'min', 'argmax', 'argmin'),
      mode=('smooth', 'c2'),
  )
  def test_operations_hard_forward_soft_gradient(self, operation, mode):
    x = jp.array([-0.05, 0.05, 0.12])

    def evaluate(x, mode, st_enable):
      kwargs = dict(mode=mode, softness=0.2, st_enable=st_enable)
      fn = getattr(sj, operation)
      if operation == 'clip':
        value = fn(x, 0.0, 1.0, **kwargs)
      elif operation in ('less', 'greater', 'greater_equal'):
        value = sj.where(fn(x, 0.0, **kwargs), 1.0, 0.0)
      elif operation in ('argmax', 'argmin'):
        index = fn(x, standardize=False, **kwargs)
        value = sj.dynamic_index_in_dim(jp.array([1.0, 3.0, -2.0]),
                                        index,
                                        keepdims=False)
      elif operation in ('max', 'min'):
        value = fn(x, standardize=False, **kwargs)
      else:
        value = fn(x, **kwargs)
      return jp.sum(value)

    hard = evaluate(x, 'hard', False)
    soft = evaluate(x, mode, False)
    st = jax.jit(lambda x: evaluate(x, mode, True))
    np.testing.assert_allclose(st(x), hard, atol=1e-6)
    self.assertFalse(np.allclose(soft, hard))
    soft_grad = jax.grad(lambda x: evaluate(x, mode, False))(x)
    st_grad = jax.jit(jax.grad(st))(x)
    hard_grad = jax.grad(lambda x: evaluate(x, 'hard', False))(x)
    np.testing.assert_allclose(st_grad, soft_grad, atol=1e-6, rtol=1e-5)
    self.assertTrue(np.all(np.isfinite(np.asarray(st_grad))))
    self.assertFalse(np.allclose(st_grad, hard_grad))

  @parameterized.product(
      helper=('segment_point', 'segment_distance', 'segment_pair',
              'plane_segment', 'manifold', 'clip', 'contact_manifold', 'box'),
      mode=('smooth', 'c2'),
  )
  def test_composite_helpers_hard_forward(self, helper, mode):
    poly = jp.array([[0., 0., 0.], [1., 0., 0.], [1., 0.98, 0.], [0., 1., 0.]])
    normal = jp.array([0., 0., 1.])
    a, b, pt = jp.array([0., 0., 0.]), jp.array([1., 0.,
                                                 0.]), jp.array([0., 0.1, 0.])

    def evaluate(offset, mode, st_enable):
      kwargs = dict(st_enable=st_enable)
      subject = poly + jp.array([offset, 0.1, 0.02])
      if helper == 'segment_point':
        result = math.closest_segment_point_soft(a, b, pt.at[0].set(offset),
                                                 mode, **kwargs)
      elif helper == 'segment_distance':
        result = math.closest_segment_point_and_dist_soft(
            a, b, pt.at[0].set(offset), mode, **kwargs)
      elif helper == 'segment_pair':
        result = math.closest_segment_to_segment_points_soft(
            a, b, pt.at[0].set(offset), jp.array([offset, 1., 0.]), mode,
            **kwargs)
      elif helper == 'plane_segment':
        result = collision_convex._closest_segment_point_plane_soft(
            a, b, pt.at[0].set(offset), jp.array([1., 0., 0.]), mode, 0.1,
            **kwargs)
      elif helper == 'manifold':
        index = collision_convex._manifold_points_soft(subject, jp.ones(4),
                                                       normal, mode, 0.1,
                                                       **kwargs)
        result = index @ subject
      elif helper == 'clip':
        result = collision_convex._clip_soft(poly, subject, normal, normal,
                                             mode, 0.1, jp.array(1.), **kwargs)
      elif helper == 'contact_manifold':
        result = collision_convex._create_contact_manifold_soft(
            poly, subject, normal, normal, normal, mode, 0.1, jp.array(1.),
            **kwargs)
      else:
        box_a = soft_collision_test._box((0., 0., 0.))
        box_b = soft_collision_test._box((offset, 0.02, 0.01))
        result = collision_convex._box_box_soft(box_a,
                                                box_b,
                                                mode,
                                                softness=0.1,
                                                **kwargs)
      return jp.concatenate([x.reshape(-1) for x in jax.tree.leaves(result)])

    offset = jp.array(0.0)
    hard = evaluate(offset, 'hard', False)
    st = jax.jit(lambda x: evaluate(x, mode, True))
    actual = st(offset)
    if helper == 'box':
      # Feature ties can enumerate the same nominal manifold differently.
      np.testing.assert_allclose(actual[:4], hard[:4], atol=1e-5)
      np.testing.assert_allclose(
          soft_collision_test._sort_rows(actual[4:16].reshape(4, 3)),
          soft_collision_test._sort_rows(hard[4:16].reshape(4, 3)), atol=1e-5)
      np.testing.assert_allclose(actual[16:], hard[16:], atol=1e-5)
    else:
      np.testing.assert_allclose(actual, hard, atol=1e-5, rtol=1e-5)
    soft = evaluate(offset, mode, False)
    self.assertTrue(np.all(np.isfinite(np.asarray(soft))))
    self.assertFalse(np.allclose(soft, hard, atol=1e-5, rtol=1e-5))
    grad = jax.jit(jax.jacrev(st))(offset)
    self.assertTrue(np.all(np.isfinite(np.asarray(grad))))


if __name__ == '__main__':
  absltest.main()

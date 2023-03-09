import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

class Bfield:
    def __init__(self, mag_moment, mag_components):
        self.moment = jnp.array(mag_moment)
        self.dip, self.quad, self.oct = mag_components

        self.vector = jax.grad(self.potential, argnums=(0,1,2))


    @jax.partial(jax.jit, static_argnums=(0,))
    def potential(self, x, y, z):
        point = jnp.array([x, y, z])
        point_distance = jnp.linalg.norm(point)
        cosine = jnp.dot(self.moment, point)/point_distance

        dip_potential = cosine / point_distance**2
        quad_potential = (1.5 * cosine**2 - 0.5) / point_distance**3
        oct_potential = (2.5 * cosine**3 - 1.5 * cosine) / point_distance**4

        total_potential = 0.5 * self.dip * dip_potential \
            + 1./3. * self.quad * quad_potential \
            + 0.25 * self.oct * oct_potential
        
        return -total_potential

    @jax.partial(jax.jit, static_argnums=(0,))
    def strength(self, x, y, z):
        point = jnp.array([x, y, z])
        return jnp.linalg.norm(self.vector(x, y, z))


def tests():
    test_point = jnp.array([0., 0., 1.])

    mag_moment = [0., 0., 1.]
    pure_dip = Bfield(mag_moment, jnp.array([1.,0.,0.]))
    pure_quad = Bfield(mag_moment, jnp.array([0.,1.,0.]))
    pure_oct = Bfield(mag_moment, jnp.array([0.,0.,1.]))

    assert pure_dip.strength(0.,0.,1.) == 8. * pure_dip.strength(0.,0.,2.)
    assert pure_quad.strength(0.,0.,1.) == 16. * pure_quad.strength(0.,0.,2.)
    assert pure_oct.strength(0.,0.,1.) == 32. * pure_oct.strength(0.,0.,2.)

    xs = jnp.linspace(-1.,1.,21)
    ys = jnp.zeros_like(xs)
    zs = jnp.linspace(-2.,2.,21)

    Xs, Zs = jnp.meshgrid(xs, zs, indexing='ij')
    assert Xs[0,0] == -1.
    assert Zs[0,0] == -2.
    assert Xs[10,5] == 0.
    assert Zs[5,5] == -1.

    bfield_vNv = jax.vmap(pure_dip.vector, in_axes=(0,None,0), out_axes=0)
    assert bfield_vNv(xs, 0.0, zs)[0][5] == pure_dip.vector(xs[5], 0.0, zs[5])[0]
    assert bfield_vNv(xs, 0.0, zs)[1][0] == pure_dip.vector(xs[0], 0.0, zs[0])[1]
    assert bfield_vNv(xs, 0.0, zs)[2][3] == pure_dip.vector(xs[3], 0.0, zs[3])[2]
    bfield_mNm = jax.vmap(bfield_vNv, in_axes=(1,None,1), out_axes=0)
    assert bfield_mNm(Xs, 0.0, Zs)[0][5][1] == pure_dip.vector(Xs[1][5], 0.0, Zs[1][5])[0]
    assert bfield_mNm(Xs, 0.0, Zs)[1][1][5] == pure_dip.vector(Xs[5][1], 0.0, Zs[5][1])[1]
    assert bfield_mNm(Xs, 0.0, Zs)[2][3][4] == pure_dip.vector(Xs[4][3], 0.0, Zs[4][3])[2]

    bfield_mNm_norms = jnp.linalg.norm(bfield_mNm(Xs, 0.0, Zs), axis=0).T
    assert bfield_mNm_norms[10][5] == pure_dip.strength(Xs[10][5], 0.0, Zs[10][5])
    
    plt.quiver(Zs, Xs, \
        bfield_mNm(Xs, 0.0, Zs)[0]/bfield_mNm_norms, bfield_mNm(Xs, 0.0, Zs)[2]/bfield_mNm_norms)
    plt.show()

    bstrength_vNv = jax.vmap(pure_dip.strength, in_axes=(0,None,0), out_axes=0)
    bstrength_mNm = jax.vmap(bstrength_vNv, in_axes=(1,None,1), out_axes=0)

    
if __name__ == "__main__":
    tests()
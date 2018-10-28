#

exec(open('common_imports.py').read())
exec(open('/home/xyy/Documents/research/poly-orth/ex/weighted_orthpoly_solver.py').read())

n_poly_order = 100
V = orthpoly_coef(None, 'legendre', n_poly_order)
x_sample = sin( pi * (-n_poly_order/2 + arange(n_poly_order+1))/(n_poly_order+1) )
get_fq_coef = lambda f: np.linalg.solve(V, f(x_sample))
#get_fq_coef = lambda f: orthpoly_coef(f, 'legendre', n_poly_order)
#get_fq_coef = lambda f: chb.Chebyshev.interpolate(f, n_poly_order).coef

cost_l = lambda pm: cost_flat(ans_xy, pm, layer_dims, active_function = 'exptip')
cost_g = ag.jacobian(cost_l)
cost_h = ag.hessian (cost_l)

# ANN function
ann_theta_x = lambda pm, x: np.squeeze(net_predict(x, wb_flat2layers(pm, layer_dims), active_function = 'exptip'))
ann_g = ag.jacobian(ann_theta_x, argnum=0)
ann_h = ag.hessian (ann_theta_x, argnum=0)

# value for parameters and input
layer_dims = [1, 2000, 1]
pm0 = 1.0 * npr.randn(n_ann_param(layer_dims))

# a glance of the ANN function
figure(11);  clf()
s_x = np.linspace(-1, 1, 100)[:, np.newaxis]
plot(s_x, ann_theta_x(pm0, s_x))

cb1 = chb.Chebyshev.interpolate(lambda x: ann_theta_x(pm0, x[:,np.newaxis]), 100)

figure(13); clf()
plt.semilogy(np.abs(cb1.coef), '-o')
plt.ylim([1e-10, 10])
pic_output('cb_coef_init', None)

def gd_callback(params, t, g):
    loss = cost_l(params)
    print("step =%4d: cost = %.5g, |g|_2 = %.3g" % \
            (t, loss, norm(g)))
    if t % 100 == 0:
        f_ann = lambda x: ann_theta_x(params, x[:,np.newaxis])
        cb1 = get_fq_coef(f_ann)
        clf()
        plt.semilogy(np.abs(cb1), '-o')
        plt.semilogy(np.abs(fun_cb), '-o')
        plt.ylim([1e-10, 10])
        plt.xlabel('k (freq. index)')
        plt.ylabel('c_k (step=%d)' % (t))
        pic_output('cb_coef_t%0.5d'%(t), None)
        clf()
        plt.plot(ans_xy['x'], ann_theta_x(params, ans_xy['x']), '-o')
        plt.plot(ans_xy['x'], ans_xy['y'], '-o')
        plt.xlabel('x')
        plt.ylabel('f(x) (step=%d)' % (t))
        pic_output('fun_t%0.5d'%(t), None)

# Function to be fitted.
fun = lambda x: (x+0.8) * np.arcsin(np.sin(2*x*x+5*x))
n_sample = 100
ans_xy = get_training_set(n_sample, fun)

fun_cb = get_fq_coef(fun)

flat_wb = pm0
n_iter = 10000
step_size = 0.0005
wb_end = adam(lambda pm, k: cost_g(pm), flat_wb, callback=gd_callback,
              num_iters=n_iter, step_size=step_size)

# vim: et sw=4 sts=4

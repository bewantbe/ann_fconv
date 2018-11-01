#

exec(open('common_imports.py').read())
exec(open('/home/xyy/Documents/research/poly-orth/ex/weighted_orthpoly_solver.py').read())

x_sample = np.linspace(-1, 1, 100)
fun = lambda x: (x+0.8) * np.arcsin(np.sin(2*x*x+5*x))
ans_xy = get_training_set(100, fun, x_sample)

n_poly_order = 100
V = orthpoly_coef(None, 'legendre', n_poly_order)
#x_sample = sin( pi * (-n_poly_order/2 + arange(n_poly_order+1))/(n_poly_order+1) )
x_sample = lege.legroots(1*(arange(n_poly_order+2)==n_poly_order+1))
get_fq_coef = lambda f: np.linalg.solve(V, f(x_sample))
#get_fq_coef = lambda f: orthpoly_coef(f, 'legendre', n_poly_order)
#get_fq_coef = lambda f: chb.Chebyshev.interpolate(f, n_poly_order).coef

#
act_fun = 'tanh'

cost_l = lambda pm: cost_flat(ans_xy, pm, layer_dims, active_function = act_fun)
cost_g = ag.jacobian(cost_l)
cost_h = ag.hessian (cost_l)

# ANN function
ann_theta_x = lambda pm, x: np.squeeze(net_predict(x, wb_flat2layers(pm, layer_dims), active_function = act_fun))
ann_g = ag.jacobian(ann_theta_x, argnum=0)
ann_h = ag.hessian (ann_theta_x, argnum=0)

# value for parameters and input
layer_dims = [1, 2000, 1]
pm0 = 1.0 * npr.randn(n_ann_param(layer_dims))

# a glance of the ANN function
#figure(11);  clf()
#s_x = np.linspace(-1, 1, 100)[:, np.newaxis]
#plot(s_x, ann_theta_x(pm0, s_x))

def gd_callback(params, t, g):
    loss = cost_l(params)
    print("step =%4d: cost = %-10.5g, |g|_2 = %.3g" % \
            (t, loss, norm(g)))
    if t % 1000 == 0:
        f_ann = lambda x: ann_theta_x(params, x[:,np.newaxis])
        cb1 = get_fq_coef(f_ann)
        clf()
        plt.semilogy(np.abs(cb1), '-o')
        plt.semilogy(np.abs(fun_cb), '-o')
        plt.ylim([1e-10, 10])
        plt.xlabel('k (freq. index)')
        plt.ylabel('c_k (step=%d)' % (t))
        pic_output('cb_coef_t%0.6d'%(t), None)
        clf()
        plt.plot(ans_xy['x'], ann_theta_x(params, ans_xy['x']), '-o')
        plt.plot(ans_xy['x'], ans_xy['y'], '-o')
        plt.xlabel('x')
        plt.ylabel('f(x) (step=%d)' % (t))
        pic_output('fun_t%0.6d'%(t), None)

# Function to be fitted.
fun = lambda x: (x+0.8) * np.arcsin(np.sin(2*x*x+5*x))
n_sample = 100
ans_xy = get_training_set(n_sample, fun)

fun_cb = get_fq_coef(fun)

flat_wb = pm0
n_iter = 1000
step_size = 1e-4
wb_end = adam(lambda pm, k: cost_g(pm), flat_wb, callback=gd_callback,
              num_iters=n_iter, step_size=step_size)

# Refine
#pm1 = wb_end
#flat_wb = pm1
#n_iter = 500000
#step_size = 1e-6
#wb_end = adam(lambda pm, k: cost_g(pm), flat_wb, callback=gd_callback,
#              num_iters=n_iter, step_size=step_size, b1=0.9, b2=0.99)

h = cost_h(wb_end)  # 105 sec for layer_dims = [1, 2000, 1]
eigval, eigvec = np.linalg.eigh(h)

figure(342); clf()
plot(log10(abs(eigval)))

if 0:
    fpath = 'pm_set/fit_1M3'
    np.savez_compressed(fpath,
        ans_xy    = ans_xy,
        layer_dims= layer_dims,
        act_fun   = act_fun,
        wb_init   = flat_wb,
        n_iter    = n_iter,
        step_size = step_size,
        wb_end    = wb_end
    )

if 0:
    # Load track path data
    fprefix = 'pm_set/'
    fname = 'fit_1M3.npz'
    with np.load(fprefix+fname) as data:
        ans_xy         = data['ans_xy'].item()
        layer_dims     = data['layer_dims']
        act_fun        = data['act_fun'].item()
        flat_wb        = data['wb_init']
        n_iter         = data['n_iter'].item()
        step_size      = data['step_size'].item()
        wb_end         = data['wb_end']

# vim: et sw=4 sts=4

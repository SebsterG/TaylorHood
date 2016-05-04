from dolfin import *
import matplotlib.pyplot as plt
import time
import numpy as np
set_log_active(False)
start_time = time.time()

N = 32
mesh = BoxMesh(Point(-pi, -pi, -pi), Point(pi, pi, pi), N, N, N)
#plot(mesh,interactive=True)

class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool((near(x[0], -pi) or near(x[1], -pi) or near(x[2], -pi)) and
                        (not (near(x[0], pi) or near(x[1], pi) or near(x[2], pi))) and on_boundary)

    def map(self, x, y):
        if near(x[0], pi) and near(x[1], pi) and near(x[2],pi):
            y[0] = x[0] - 2.0*pi
            y[1] = x[1] - 2.0*pi
            y[2] = x[2] - 2.0*pi
        elif near(x[0], pi) and near(x[1], pi):
            y[0] = x[0] - 2.0*pi
            y[1] = x[1] - 2.0*pi
            y[2] = x[2]
        elif near(x[1], pi) and near(x[2], pi):
            y[0] = x[0]
            y[1] = x[1] - 2.0*pi
            y[2] = x[2] - 2.0*pi
        elif near(x[1], pi):
            y[0] = x[0]
            y[1] = x[1] - 2.0*pi
            y[2] = x[2]
        elif near(x[0], pi) and near(x[2], pi):
            y[0] = x[0] - 2.0*pi
            y[1] = x[1]
            y[2] = x[2] - 2.0*pi
        elif near(x[0], pi):
            y[0] = x[0] - 2.0*pi
            y[1] = x[1]
            y[2] = x[2]
        else:
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - 2.0*pi
V = VectorFunctionSpace(mesh, "CG", 2, constrained_domain=PeriodicBoundary())
Q = FunctionSpace(mesh,"CG", 1,constrained_domain=PeriodicBoundary())
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

PB = PeriodicBoundary()
bound = FacetFunction("size_t", mesh)
bound.set_all(0)
PB.mark(bound,1)
n = FacetNormal(mesh)
#plot(bound,interactive=True)


nu = 1.0/1000.0 # Re = 1600
p_0=Expression('1./16.*(cos(2*x[0])+cos(2*x[1]))*(cos(2*x[2])+2)')
u0 = project(Expression(('sin(x[0])*cos(x[1])*cos(x[2])','-cos(x[0])*sin(x[1])*cos(x[2])',"0")),V)
#print "norm: ",norm(u0)
#plot(u0)#,interactive=True)

bcs=[]
bcp=[]


u1 = Function(V)
u_star = Function(V)
p0 = Function(Q)
p1 = Function(Q)

dt = 1.0/1000.0
rho = Constant(1)
nu = Constant(nu)
K = Constant(dt)
def sigma(u, p):
    return 2.0*nu*sym(grad(u))-p*Identity(len(u))
def eps(u):
    return sym(grad(u))

f = Constant((0.0,0.0,0.0))

a1 = (1.0/K)*rho*inner(u-u0,v)*dx + \
inner(grad(u0)*u0, v)*dx+\
inner(sigma(0.5*(u+u0),p0),eps(v))*dx \
- nu*dot(dot(grad(0.5*(u+u0)),n),v)*ds(1) + \
dot(p0*n,v)*ds(1) - inner(f,v)*dx

a2 = K *inner(grad(p),grad(q))*dx
L2 = K *inner(grad(p0),grad(q))*dx - rho*inner(div(u_star),q)*dx

a3 = rho*inner(u,v)*dx
L3 = rho*inner(u_star,v)*dx - K*inner(grad(p1-p0),v)*dx

A1 = assemble(lhs(a1)); A2 = assemble(a2); A3 = assemble(a3)
b1 = None; b2 = None; b3 = None


e_k = []; dKdt = []; time_array = []
#curlfile = File("curl.pvd")
T = 20.0
t = dt
counter = 0
while t < T + DOLFIN_EPS:
    # Update pressure boundary condition
    #solve(a1==L1, u_star,bcs)
    b1 = assemble(rhs(a1),tensor = b1)
    [bc.apply(A1,b1) for bc in bcs]
    pc = PETScPreconditioner("jacobi")
    sol = PETScKrylovSolver("bicgstab", pc)
    sol.solve(A1, u_star.vector(), b1)

    #solve(A1,u_star.vector(),b1,"bicgstab","pc")
    #pressure correction
    #solve(a2==L2,p1,bcp,solver_parameters={"linear_solver":"gmres"})

    b2 = assemble(L2,tensor = b2)
    [bc.apply(A2,b2) for bc in bcp]
    solve(A2,p1.vector(),b2,"gmres","hypre_amg")
    #print norm(p1)

    #last step
    b3 = assemble(L3,tensor=b3)
    [bc.apply(A3,b3) for bc in bcs]
    pc2 = PETScPreconditioner("jacobi")
    sol2 = PETScKrylovSolver("cg", pc2)
    sol2.solve(A3,u1.vector(),b3)

    u0.assign(u1)
    p0.assign(p1)
    plot(u1)

    print "Timestep: ", t
    if (counter%100==0 or counter%100 == 1):
        kinetic_e = assemble(0.5*dot(u1,u1)*dx)/(2*pi)**3
        if (counter%100)==0:
            kinetic_hold = kinetic_e
            #dissipation_e = assemble(nu*inner(grad(u1), grad(u1))*dx) / (2*pi)**3
            #print "dissipation: ", dissipation_e
            print "kinetic energy", kinetic_e
        else: # (counter%100)==1:
            if MPI.rank(mpi_comm_world())==0:
                e_k.append((kinetic_e))
                dKdt.append(-(kinetic_e - kinetic_hold)/dt)
                time_array.append(t)
            #    curlfile << project(curl(u1)[2],Q)
    #plot(p1,rescale=True)
    counter+=1
    t += dt

print("--- %s seconds ---" % (time.time() - start_time))
if MPI.rank(mpi_comm_world())==0:
    np.savetxt('results/IPCS/dKdt_32.txt', dKdt,delimiter =',')
    np.savetxt('results/IPCS/e_k_32.txt', e_k, delimiter = ',')
    np.savetxt("results/IPCS/time_32.txt",time, delimiter = "," )

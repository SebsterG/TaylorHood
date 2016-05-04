from dolfin import *
import matplotlib.pyplot as plt
import time
import numpy as np
set_log_active(False)
start_time = time.time()

N = 10
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
#plot(bound,interactive=True)


nu = 1.0/1000.0 # Re = 1600
p_0=Expression('1./16.*(cos(2*x[0])+cos(2*x[1]))*(cos(2*x[2])+2)')
u0 = project(Expression(('sin(x[0])*cos(x[1])*cos(x[2])','-cos(x[0])*sin(x[1])*cos(x[2])',"0")),V)
#print "norm: ",norm(u0)
#plot(u0)#,interactive=True)
u1 = Function(V)
p1 = Function(Q)

bcs=[]
bcp=[]

dt = 0.001

k = Constant(dt)
f = Constant((0.0, 0.0,0.0))
nu = Constant(nu)
# first without Pressure
F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx +  nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# correction with Pressure
a2 = -k*inner(grad(p),grad(q))*dx
L2 = div(u1)* q *dx

# last step

a3 = inner(u,v)*dx
L3 = inner(u1,v)*dx - k*inner(grad(p1),v)*dx

A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

#ufile = File("results/velocity.pvd")
#pfile = File("results/pressure.pvd")
#curlfile = File("results/curl.pvd")

T = 20
t = dt
counter = 0
dKdt = [];e_k = [];time_array = [];diss = []
b1 = None ; b2 = None; b3 = None
while t < T + DOLFIN_EPS:
    # Update pressure boundary condition
    # solve(a1==L1,u1,bcs,solver_parameters={"linear_solver": "gmres"})
    b1 = assemble(L1,tensor=b1)
    [bc.apply(A1,b1) for bc in bcs]
    pc = PETScPreconditioner("jacobi")
    sol = PETScKrylovSolver("bicgstab",pc)
    sol.solve(A1,u1.vector(),b1)
    #pressure correction
    #solve(a2==L2,p1,bcp,solver_parameters={"linear_solver":"gmres"})

    b2 = assemble(L2,tensor=b2)
    [bc.apply(A2,b2) for bc in bcp]
    solve(A2,p1.vector(),b2,"gmres","default")
    #print norm(p1)

    #last step
    #solve(a3==L3,u1,bcs,solver_parameters={"linear_solver":"gmres"})
    b3 = assemble(L3,tensor=b3)
    [bc.apply(A3,b3) for bc in bcs]
    pc2 = PETScPreconditioner("jacobi")
    sol2 = PETScKrylovSolver("cg",pc2)
    solve(A3,u1.vector(),b3)

    u0.assign(u1)
    #print MPI.rank(mpi_comm_world())


    print "Timestep: ", t
    if (counter%100==0 or counter%100 == 1):
        kinetic_e = assemble(0.5*dot(u1,u1)*dx)/(2*pi)**3
        if (counter%100)==0:
            kinetic_hold = kinetic_e
            dissipation_e = assemble(nu*inner(grad(u1), grad(u1))*dx) / (2*pi)**3
            print "dissipation: ", dissipation_e
            print "kinetic energy", kinetic_e
        else: # (counter%100)==1:
            if MPI.rank(mpi_comm_world())==0:
                e_k.append((kinetic_e))
                dKdt.append(-(kinetic_e - kinetic_hold)/dt)
                time_array.append(t)
                #curlfile << project(curl(u1)[2],Q)
    #plot(p1,rescale=True)
    counter+=1
    t += dt

print("--- %s seconds ---" % (time.time() - start_time))

np.savetxt('results/chorin/dKdt_10.txt', dKdt, delimiter=',')
np.savetxt('results/chorin/E_k_10.txt', e_k, delimiter=',')
np.savetxt('results/chorin/time._10txt',time_array , delimiter=',')

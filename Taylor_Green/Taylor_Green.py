from dolfin import *
set_log_active(False)
N = 10
mesh = UnitSquareMesh(N,N)
V = VectorFunctionSpace(mesh,"CG", 2)
Q = FunctionSpace(mesh,"CG", 1)
#VQ = V * Q # The Mixed space , alternative writing :
#VQ = M i x e d F u n c t i o n S p a c e ([V , Q])
#u , p = TrialFunctions(VQ)
#v , q = TestFunctions(VQ)
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

class Circle(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and not (near(x[0],0) or near(x[0],2.2) or near(x[1],0) or near(x[1],0.41))

class Right(SubDomain):
	def inside(self,x,on_boundary):
		return near(x[0], 2.2)

class Left(SubDomain):
	def inside(self,x,on_boundary):
		return near(x[0], 0)
class Nos(SubDomain):
	def inside(self,x,on_boundary):
		return near(x[1], 0) or near(x[1],0.41)


circle = Circle()
left = Left()
right = Right()
nos = Nos()
bound = FacetFunction("size_t", mesh)
bound.set_all(0)
nos.mark(bound, 3)
circle.mark(bound, 1)
left.mark(bound,4)
right.mark(bound,2)
#plot(bound); interactive()
Um = 0.3
H = 0.41
inlet = Expression(("4*Um*x[1]*(H-x[1])/(H*H)", "0"),t=0.0,Um = Um,H=H)
#U_mean = project(Expression(("4*Um*(H/2.0)*(H/2.0)/(H*H)","0"),Um=Um,H=H),V)
U_mean = 4*Um*(H/2.0)*(H/2.0)/(H*H)
v_theta = Expression(("0","0"))

bc1 = DirichletBC(V, inlet , left)
bc2 = DirichletBC(V, (0, 0), nos)
bc3 = DirichletBC(V, v_theta , circle)
bc4 = DirichletBC(Q, 0.0,right)
bcs = [bc1,bc2,bc3]
bcp = [bc4]

u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)

dt = 0.01
nu = 0.001

k = Constant(dt)
f = Constant((0, 0))

# first without Pressure
#F1 = (1/k)*inner(u-u0,v)*dx + inner(dot(grad(u0),u0),v)*dx + nu*inner(grad(u),grad(v))*dx - inner(f, v)*dx
F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx +  nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# correction with Pressure
a2 = -k*inner(grad(p),grad(q))*dx
L2 = div(u1)* q *dx

# last step

a3 = inner(u,v)*dx
L3 = inner(u1,v)*dx - k*inner(grad(p1),v)*dx


#ufile = File("results/velocity.pvd")
#pfile = File("results/pressure.pvd")

T = 10.0
t = dt
counter = 0
while t < T + DOLFIN_EPS:
	# Update pressure boundary condition
	inlet.t = t
	#v_theta.t = t
	solve(a1==L1,u1,bcs)

	#pressure correction
	solve(a2==L2,p1,bcp)
	#print norm(p1)

	#last step
	solve(a3==L3,u1,bcs)

	u0.assign(u1)
	t += dt
	print "Timestep: ", t

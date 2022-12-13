import numpy as np
import matplotlib.pyplot as plt
from rotational_diffusion.fluorophore_rotational_diffusion import Fluorophores, FluorophoreStateInfo
from numpy.random import uniform

# all written by Julia

# Note that diffusion times used here are not using the same convention
# as Lackowicz and the field of fluorescence anisotropy. The
# diffusion_time here is a factor of pi larger than the rotational
# correlation time we calculate in anisotropy measurements. If you have
# a rotational correlation time of, say, 100 ns, you should enter 314 ns
# for the diffusion time.

#### DEMO 1. Simulating fluorescence anisotropy
#### Note that a test function for this already exists in the imported code,
#### but I'm including this here to demonstrate plot functionality.
##print("Simulating classic anisotropy decay...", sep='', end='')
##f = Fluorophores(1e8, diffusion_time=1)
##f.phototransition('ground', 'excited',
##                  intensity=0.05, polarization_xyz=(1,0,0))
##while f.orientations.n > 0:
##    print('.', sep='', end='')
##    f.delete_fluorophores_in_state('ground')
##    f.time_evolve(0.3)
##print("done.")
##x, y, z, t = f.get_xyzt_at_transitions('excited', 'ground')
### The probability of landing in a channel depends on the cosine^2 of the
### angle between the two unit vectors (in this case, the vector fully in,
### say, the x direction, and the vector of the molecule's orientation at
### the time of emission). This is basically the dot product formula
### rearranged & squared, with the terms that are 0 dropped out.
##p_x, p_y = x**2, y**2 # Probabilities of landing in channel x or y
##r = uniform(0, 1, size=len(t))
##in_channel_x = (r < p_x)
##in_channel_y = (p_x <= r) & (r < p_x + p_y)
##t_x, t_y = t[in_channel_x], t[in_channel_y]
##bins = np.linspace(0, 3, 200)
##bin_centers = (bins[1:] + bins[:-1])/2
##(hist_x, _), (hist_y, _) = np.histogram(t_x, bins),  np.histogram(t_y, bins)
##
##plt.figure()
##plt.plot(bin_centers, hist_x, '.-', label=r'$\parallel$ polarization')
##plt.plot(bin_centers, hist_y, '.-', label=r'$\perp$ polarization')
##plt.title(
##    "Simulation of classic time-resolved anisotropy decay\n" +
##    r"$\tau_f$=%0.1f, $\tau_d$=%0.1f"%(f.state_info['excited'].lifetime,
##                                       f.orientations.diffusion_time))
##plt.xlabel(r"Time (t/$\tau_f$)")
##plt.ylabel("Photons per time bin")
##plt.legend(); plt.grid('on')
##plt.savefig("test_classic_anisotropy_decay.png"); plt.close()

###############################################################################
###############################################################################
###############################################################################
def make_animation_frame(my_fluorophores, filename, frame_number,
                         view_angle=(10,45), state=None, label=None):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1, projection='3d')
    if state == None: # default, plot all of the molecules
        o = my_fluorophores.orientations # nickname
        x = o.x; y=o.y; z=o.z
    else: # only plot a subset of the molecules
        x, y, z = my_fluorophores.get_xyz_for_state(state)    
    ax.scatter(x, y, z,
               c=np.linspace(0.35, 0.85, len(x)),
               cmap=plt.cm.inferno, vmin=0, vmax=1)
    ax.view_init(*view_angle)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_box_aspect((1, 1, 1))
    ax.xaxis._axinfo["grid"].update({"color":'#dcdcdc'})
    ax.yaxis._axinfo["grid"].update({"color":'#dcdcdc'})
    ax.zaxis._axinfo["grid"].update({"color":'#dcdcdc'})
    if label is not None: plt.suptitle(label)
    plt.savefig('%s_frame_%06i.png'%(filename, frame_number),
                bbox_inches='tight', dpi=100)
    plt.close(fig)
    
### Let's do another anisotropy simulation to demonstrate the "points on a
### sphere" visualization of traditional anisotropy with orthogonal STED depletion
### A similar animation of rotational diffusion, but now with initial
### "crescent selection" via activation and saturated depletion.
##print("\Animation 2...")
##state_info = FluorophoreStateInfo()
##state_info.add('ground')
##state_info.add('excited', lifetime=1, final_states='ground')
##a = Fluorophores(3e4,
##                 diffusion_time=5,
##                 state_info=state_info)
### Plot the initial fluorophores
##current_frame = 0
##make_animation_frame(a, 'Animation/animation_2', current_frame)
##current_frame += 1
##
### Photoselect in Z (traditional anisotropy)
##a.phototransition('ground',
##                  'excited',
##                  intensity=0.5,
##                  polarization_xyz=(0, 0, 1))
##a.delete_fluorophores_in_state('ground')
##make_animation_frame(a, 'Animation/animation_2', current_frame)
##current_frame += 1
##
### Deactivate in X, saturating (STED anisotropy)
##a.phototransition('excited',
##                  'ground',
##                  intensity=100,
##                  polarization_xyz=(1, 0, 0))
##a.delete_fluorophores_in_state('ground')
##make_animation_frame(a, 'Animation/animation_2', current_frame)
##current_frame += 1
##
##for i in range(10):
##    a.time_evolve(0.001)
##    a.delete_fluorophores_in_state('ground')
##    make_animation_frame(a, 'Animation/animation_2', current_frame)
##    current_frame += 1

## Now, let's do an animation of triplet state photophysics. We are
## going to generate triplets and then trigger them with an orthogonally
## polarized beam. For now, let's just look at what distribution
## crescent selection of triplets gives. We are going to interleave
## pulses of triplet generation and triplet triggering every 12.5 ns
## (this might even be accurate re: the SP8 experiments)

print("Animation 3: Triggered Triplets")
output_name = 'animation_1'
state_info = FluorophoreStateInfo()
state_info.add('ground')
state_info.add('excited_singlet', lifetime=4,
               final_states=['ground', 'excited_triplet'],
               probabilities=[0.99, 0.01])
state_info.add('excited_triplet',
               lifetime=5e5, # mScarlet 500 us
               final_states='ground')
a = Fluorophores(1e6,
                 diffusion_time=1.6e5*np.pi, # 100nm~160 us, 50nm~20 us (Ilaria)
                 state_info=state_info)

print('Generating photoactivated triplets', end='')
# Let's photoactivate (& generate an animation of what this looks like)
current_frame = 0
for i in range(16): #80 MHz, 200 ns (actual SP8 parameters)
    print('.', end='')
    # Generate triplets in Z 
    a.phototransition('ground',
                      'excited_singlet',
                      intensity=0.5,
                      polarization_xyz=(0, 0, 1))
    a.time_evolve(2)
    # Crescent selection of triplets with the STED beam in X
    a.phototransition('excited_triplet',
                      'excited_singlet',
                      intensity=3, # empirical, 25% STED is ~90% saturating
                      polarization_xyz=(1, 0, 0))
    a.time_evolve(10.5)
    make_animation_frame(a, output_name, current_frame, state='excited_singlet',
                         label='singlet', view_angle=(10, 45))
    current_frame += 1
    make_animation_frame(a, output_name, current_frame, state='excited_triplet',
                         label='triplet', view_angle=(10, 45))
    current_frame += 1

print('\nTime evolving', end='')
# Now, let's time evolve a long way and see what we'd expect from an experiments
a.delete_fluorophores_in_state('ground') # performance
for x in range(1000): # sort of a time resolved experiments
    a.time_evolve(5) # going in 5 ns chunks
    a.delete_fluorophores_in_state('ground')
    a.phototransition('excited_triplet',
                      'excited_singlet',
                      intensity=0.05, #lower STED so we don't deplete everything
                      polarization_xyz=(1, 0, 0))
    print('.', end='')

x, y, z, t = a.get_xyzt_at_transitions('excited_singlet', 'ground')
# Let's make a plot of the triplet anisotropy signal
p_x, p_z = x**2, z**2 # Probabilities of landing in channel x or z
r = uniform(0, 1, size=len(t))
in_channel_x = (r < p_x)
in_channel_z = (p_x <= r) & (r < p_x + p_z)
t_x, t_z = t[in_channel_x], t[in_channel_z]
bins = np.linspace(0, 5250, 50)
bin_centers = (bins[1:] + bins[:-1])/2
(hist_x, _), (hist_z, _) = np.histogram(t_x, bins),  np.histogram(t_z, bins)

plt.figure()
plt.plot(bin_centers, hist_x, '.-', label=r'$\parallel$ to 775')
plt.plot(bin_centers, hist_z, '.-', label=r'$\perp$ to 775')
plt.xlabel("Time (ns)")
plt.ylabel("Photons per time bin")
plt.legend(); plt.grid('on')
plt.savefig("crescent_triggered_triplets.png")
plt.show()




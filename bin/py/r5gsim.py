
from saic5g.simulators.R5GSim import ransimulator5g as simulator
import hydra
from omegaconf import DictConfig
import warnings

warnings.filterwarnings("ignore")

@hydra.main(config_path="../../configs/hydra/", config_name="r5gsim", version_base=None)
def run_sim(cfg : DictConfig):
    sim = simulator.RANSimulator5g(cfg, 'circle')
    sim.create_simulation()
    sim_iterations = 3600*24
    for it in range(sim_iterations):
        sim.simulate()
        if it%1 == 0:
            for i, c in enumerate(sim.sites):
                print('Site'+str(i), round(c.get_load(), 2), end=', ')
            print()
        if it%200 == 0:
            print('\nProgress', round((it*100/sim_iterations), 2),'%')


run_sim()

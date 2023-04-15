import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import numpy as onp
from . import helpers
import collections
import jax
from numpyro.infer.autoguide import AutoDelta
import jax.random as random
from numpyro.infer import Predictive

base_key = random.PRNGKey(1)

class RandomKeyGenerator:
    def __init__(self,seed):
        self.key=random.PRNGKey(seed)

    def get_key(self):
        self.key,key = random.split(self.key)
        return key
          
key_generator = RandomKeyGenerator(22)

def calc_dates(branch_lengths_array,root_date,config):
    calc_dates = helpers.do_branch_matmul(
            config["rows"],
            config["cols"],
            branch_lengths_array=branch_lengths_array,
            final_size=config["size"])
    return calc_dates + root_date

def calc_deltas(deltas,root_delta,config):
        do_all_branch_matmul = jax.vmap(lambda x: helpers.do_branch_matmul(
            config["rows"],
            config["cols"],
            branch_lengths_array=x,
            final_size=config["size"]),1)
        locations = do_all_branch_matmul(deltas)
        return locations.T+root_delta


class Logger:
    def __init__(self,guide):
          self.guide=guide
    def sample_variables(self,params):
        predictive = Predictive(self.guide, params=params, num_samples=1)
        all_samples = predictive(key_generator.get_key()) 
        sample =dict(
            zip(all_samples.keys(),[all_samples[key][0] for key in all_samples.keys()])
        )
        return sample
    
    
    

class BranchModelLogger(Logger):
    def __init__(self,guide,data):
        self.tree_matrix_config={"rows":data['rows'],"cols":data["cols"],"size":data["terminal_target_dates_array"].shape[0]}
        self.terminal_target_dates_array = data["terminal_target_dates_array"]
        self.branch_distances_array = data["branch_distances_array"]
        super().__init__(guide)
    
    def extract_variables(self,sample):
        results = collections.OrderedDict()
        times = self.get_branch_times(sample)
        new_dates = calc_dates(times, sample['root_date'],self.tree_matrix_config)
        results['date_cor'] = onp.corrcoef(self.terminal_target_dates_array,
                                           new_dates)[0, 1]
        results['date_error'] = onp.mean(
            onp.abs(self.terminal_target_dates_array -
                    new_dates))  # Average date error should be small
        results['date_error_med'] = onp.median(
            onp.abs(self.terminal_target_dates_array -
                    new_dates))  # Average date error should be small

        results['max_date_error'] = onp.max(
            onp.abs(self.terminal_target_dates_array - new_dates)
        )  # We know that there are some metadata errors, so there probably should be some big errors
        results['length_cor'] = onp.corrcoef(
            self.branch_distances_array,
            times)[0, 1]  # This correlation should be relatively high

        results['root_date'] = sample['root_date']
        results['mutation_rate'] = sample['latent_mutation_rate']
        return results

    def get_logging_results(self,params):
        sample = super().sample_variables(params)
        results = self.extract_variables(sample)
        return results
    
    def set_branch_lengths(self,params,tree,names_init,**kwargs):
        sample = super().sample_variables(params)
        return self._set_branch_lengths(sample,tree,names_init,**kwargs) 

    def _set_branch_lengths(self,sample,tree,names_init,**kwargs):
        branch_length_lookup = dict(
            zip(names_init,
                self.get_branch_times(sample)))
        

        total_lengths_in_time = {}

        total_lengths = dict()

        for i, node in enumerate(helpers.preorder_traversal(tree.root)):

            if not node.label:
                node_name = helpers.get_unnnamed_node_label(i)
                if kwargs["name_all_nodes"]:
                    node.label = node_name
            else:
                node_name = node.label.replace("'", "")
            node.edge_length = branch_length_lookup[node_name] / (
                365 if kwargs["output_unit"] == "years" else 1)
 
            if not node.parent:
                total_lengths[node] = branch_length_lookup[node_name]
            else:
                total_lengths[node] = branch_length_lookup[
                    node_name] + total_lengths[node.parent]

            if node.label:
                total_lengths_in_time[node.label.replace(
                    "'", "")] = total_lengths[node]
                
        return total_lengths_in_time
        
    def set_node_attributes(self,params,tree,names_init,**kwargs):
        return None

    def get_branch_times(self, sample):
        return sample['latent_time_length']
         
class LocationModelLogger(Logger):
    def __init__(self,guide,data):
        self.tree_matrix_config={"rows":data['rows'],"cols":data["cols"],"size":data["terminal_target_locations"].shape[0]}
        self.terminal_target_locations = data["terminal_target_locations"]
        self.origin = data["origin"]
        super().__init__(guide)
    
    def extract_variables(self,sample):
        results = collections.OrderedDict()
            
        deltas = sample["location_deltas"]
        root_delta = sample["location_root_delta"]
        new_traits = calc_deltas(deltas, root_delta,self.tree_matrix_config)

        
        results['max_lat_error'] = onp.max(
            onp.abs(self.terminal_target_locations[:,0] -
                    new_traits[:,0]))

        results['max_lon_error']=onp.max(
            onp.abs(self.terminal_target_locations[:,1] -
                    new_traits[:,1]))
        results['lat_error'] = onp.mean(
            onp.abs(self.terminal_target_locations[:,0] -
                    new_traits[:,0]))

        results['lon_error']=onp.mean(
            onp.abs(self.terminal_target_locations[:,1] -
                    new_traits[:,1]))
        
        # TODO make this the root location. currently it is not. it is part of the delta from the origin
        # results["root_lat"] = root_delta[0]
        # results["root_lon"] = root_delta[1]

        return results
        
    def get_logging_results(self,params):
        sample = super().sample_variables(params)
        results = self.extract_variables(sample)
        return results
    
    def set_branch_lengths(self,**kwargs):
        return None
    
    def set_node_attributes(self,params,tree,names_init,**kwargs):
        sample = super().sample_variables(params)
        return self._set_node_attributes(sample,tree,names_init,**kwargs)
    
    def _set_node_attributes(self,sample,tree,names_init,**kwargs):
        
        total_trait_deltas=dict()
        final_trait_lookup=dict(zip(names_init,
            sample['location_deltas']
        ) )

        for i, node in enumerate(helpers.preorder_traversal(tree.root)):
            if not node.label:
                node_name = helpers.get_unnnamed_node_label(i)
                if kwargs["name_all_nodes"]:
                    node.label = node_name

            else:
                node_name = node.label.replace("'", "") 
            trait_string=[]
            if not node.parent:
                final_trait = final_trait_lookup[node_name]+ self.origin + sample['location_root_delta']
                total_trait_deltas[node]=final_trait
                trait_string.append(f"location={{{','.join(map(lambda x: str(x),final_trait))}}}")
            else:
                final_trait = final_trait_lookup[node_name]+ total_trait_deltas[node.parent]
                total_trait_deltas[node]= final_trait
                trait_string.append(f"location={{{','.join(map(lambda x: str(x),final_trait))}}}")
            if len(trait_string)>0:
                    base_name= "" if not node.label else node.label
                    node.label=base_name + f'[&{",".join(trait_string)}]'

        return total_trait_deltas


class CombinedModelLogger(Logger):
    def __init__(self,guide,data):
        self.branchLogger = BranchModelLogger(guide,data["branchModel"])
        self.locationLogger = LocationModelLogger(guide,data["locationModel"])

        super().__init__(guide)
    
    def extract_variables(self,sample):
        branch_variables = self.branchLogger.extract_variables(sample)
        location_variables = self.locationLogger.extract_variables(sample)
        return collections.OrderedDict({**branch_variables, **location_variables})
        
    def get_logging_results(self,params):
        sample = super().sample_variables(params)
        results = self.extract_variables(sample)
        return results
    
    def set_branch_lengths(self,params,tree,names_init,**kwargs):
        sample = super().sample_variables(params)
        return self.branchLogger._set_branch_lengths(sample,tree,names_init,**kwargs)
    
    def set_node_attributes(self,params,tree,names_init,**kwargs):
        ##TODO don't like this is independent from above, but fine for now with delta guide
        sample = super().sample_variables(params)
        return self.locationLogger._set_node_attributes(sample,tree,names_init,**kwargs)


def branchModel(config,data):
    root_date = numpyro.sample("root_date",
                                   dist.Normal(loc=0.0, scale=1000.0))

    branch_times = numpyro.sample(
            "latent_time_length",
            dist.Uniform(
                low=onp.ones(data["branch_distances_array"].shape[0]) * 0,
                high=onp.ones(data["branch_distances_array"].shape[0]) * 365 *
                10000))

    if config["enforce_exact_clock"]:
            mutation_rate = config["clock_rate"]
    else:
        mutation_rate = numpyro.sample(
                f"latent_mutation_rate",
                dist.Uniform(
                    low=0.0,
                    high=config["clock_rate"] * 1000.0))

    branch_distances = numpyro.sample("branch_distances",
                                          dist.Poisson(mutation_rate *
                                                       branch_times / 365),
                                          obs=data["branch_distances_array"])

    calced_dates = calc_dates(branch_times, root_date,{"rows":data["rows"],"cols":data["cols"],"size":data["terminal_target_errors_array"].shape[0]})

    final_dates = numpyro.sample(f"final_dates",
                                     dist.Normal(
                                         calced_dates, config["variance_dates"] *
                                         data["terminal_target_errors_array"]),
                                     obs=data["terminal_target_dates_array"])
    return branch_times




def locationModel(config,data,branch_times):

    location_dim = data["terminal_target_locations"][0].shape[0]



    sigma_BD = numpyro.sample("location_sigma_BD", dist.Exponential(jnp.ones(location_dim))) 
    rho = numpyro.sample("location_Rho", dist.LKJ(location_dim, config["LKJ_concentration"])) 
    cov =  jnp.outer(sigma_BD, sigma_BD) * rho

    tree_length = jnp.sum(branch_times)
    scale = branch_times/(tree_length) # scaled by time with reciprocal branch rates 

    if(config['RRW']):
        branch_rates = numpyro.sample("location_branch_rates",dist.Gamma(jnp.ones(branch_times.shape[0])*0.5,jnp.ones(branch_times.shape[0])*0.5))
        scale=scale/branch_rates

        # t = numpyro.sample("t",dist.LogNormal)

    extended_dimensions = tuple([i for i in range(-1,(location_dim*-1)-1,-1) ])
    scale = jnp.expand_dims(scale,extended_dimensions)
    
    root_delta = numpyro.sample(
        "location_root_delta", dist.MultivariateNormal(jnp.stack(jnp.zeros(location_dim)), config["root_location_scale"]*cov)) #dist.MultivariateNormal(jnp.array([[0,0]]), cov*jnp.array([0,1])[:,jnp.newaxis,jnp.newaxis]).sample(random.PRNGKey(0))
    ## plate for multiple traits?

    
    location_delta = numpyro.sample(
        "location_deltas", dist.MultivariateNormal(jnp.stack(jnp.zeros(location_dim)), cov*scale)) # one more newaxis than number of traits print(cov*jnp.expand_dims(jnp.array([0,1]),(-1,-2,-3))) or print(cov*jnp.array([0,1])[:,jnp.newaxis,jnp.newaxis,jnp.newaxis])

    calced_deltas = calc_deltas(location_delta,root_delta,{"rows":data["rows"],"cols":data["cols"],"size":data["terminal_target_locations"].shape[0]})

    calced_lat_log_error = numpyro.sample(
        "final_location", 
        dist.MultivariateNormal(calced_deltas, config["sampling_covariance"]), # could include multiplier here for precision of measurements
                                obs=data["terminal_target_locations"])

def combined_model(config,data):
    branch_times=branchModel(config["branchModel"],data["branchModel"])
    locationModel(config['locationModel'],data['locationModel'],branch_times)
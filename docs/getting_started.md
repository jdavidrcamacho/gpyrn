Using `gpyrn` should be simple if you are familiar with Python. Just import the
package directly or each of the three sub-packages

```python
import gpyrn

from gpyrn import kernels, means, variational
```

The `kernels` module provides covariance functions to be used for the
GPRN nodes and weights. `means` provides the mean functions to use for a
given dataset. Note that, in the GPRN model, the nodes and weights are
independent GPs with mean zero; these mean functions will apply to the output
datasets. The heavy-lifting is done by the mean-field approximation that is
implemented in `variational`.

As described in the [examples](/examples), the typical use will be to
instantiate a `variational.MeanFieldInference` object passing in the observed datasets, and
then defining the GPRN components (nodes, weights, and means). So typically you
would do something like


```python
# load data...

# create an inference object
gprn = variational.MeanFieldInference(N_NODES, time_array, *outputs_and_errors)

# define GPRN components
nodes = [
    kernels. ...
]
weights = [
    kernels. ...
]

means = [
    means. ...
]

jitters = [...]

gprn.set_components(nodes, weights, means, jitters)
```

after which you can calculate `gprn.ELBO` or optimize the parameters with
`gprn.optimize()`.

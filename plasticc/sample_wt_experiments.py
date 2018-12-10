from plasticc.constants import DEFAULT_SAMPLE_WEIGHTS, CLASS_WEIGHTS

config0 = {53: 2, 52: 2, 90: .5}
config1 = {53: 3, 52: 3, 90: .33}
config2 = {53: 3, 52: 3, 90: .33, 6:3, 64: 3, 95:3, 90: .33}
config3 =  {53: 3, 52: 3, 90: .33, 6:3, 64: 3, 95:3, 90: .33, 64:2, 15:2}
CONFIGS_DEC7 = [config1, config2, config3]


def gen_sw_experiments(configs):
    weights = []
    for c in configs:
        sw = DEFAULT_SAMPLE_WEIGHTS.copy()
        cw = CLASS_WEIGHTS.copy()
        for k,v in c.items():
            sw[k] = sw[k] * v
            if cw[k] == 1:
                cw[k] = cw[k] * v
        weights.append(dict(sweights=sw, class_weights=cw))
    return weights

from plasticc.constants import DEFAULT_SAMPLE_WEIGHTS, CLASS_WEIGHTS

config0 = {53: 2, 52: 2, 90: .5}
config1 = {53: 3, 52: 3, 90: .33}
config2 = {53: 3, 52: 3, 90: .33, 6:3, 64: 3, 95:3, 90: .33}
config3 =  {53: 3, 52: 3, 90: .33, 6:3, 64: 3, 95:3, 64:2, 15:2}
config4 =  {53: 4, 52: 4, 90: .33, 6:4, 64: 4, 95:5,  64:4, 15:2}
config5 =  {53: 5, 52: 5, 90: .33, 6:5, 64: 5, 95:5,  64:5, 15:2}
config6 =  {53: 10, 52: 5, 90: .33, 6:10, 64: 5, 95:5,  64:10, 15:2}
config7 =  {53: 20, 52: 5, 90: .33, 6:20, 64: 5, 95:5,  64:20, 15:2}
config8 =  {53: 30, 52: 5, 90: .33, 6:30, 64: 5, 95:5,  64:30, 15:2}
config9 =  {53: 15, 52: 5, 90: .33, 6:15, 64: 5, 95:5,  64:15, 15:2}
config10 =  {53: 25, 52: 5, 90: .33, 6:25, 64: 5, 95:5,  64:25, 15:2}


CONFIGS_DEC7 = [config1, config2, config3]

def gen_sw(config4):
    sw4 = gen_sw_experiments([config4])[0]['sweights']
    return sw4

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

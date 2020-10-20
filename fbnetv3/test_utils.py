search_space = {
    'stage1': {
        '_type': 'layer_choice',
        '_value': [
            ('Conv', {
                'k': [3],
                'e': None,
                'c': (16, 24, 2),
                's': [1],
                'se': None,
            }),
            ('MBConv', {
                'k': [3, 5],
                'e': (1, ),
                'c': (16, 24, 2),
                's': [1],
                'se': [False],
            }),
            ('MBConv', {
                'k': [3, 5],
                'e': (4, 7),
                'c': (20, 32, 4),
                's': [1],
                'se': [True],
            }),
            ('MBConv', {
                'k': [3, 5],
                'e': (4, 7),
                'c': (24, 48, 4),
                's': [1],
                'se': [True],
            })
        ]
    },
    'stage2': {
        '_type': 'layer_choice',
        '_value': [
            ('Conv', {
                'k': [3],
                'e': None,
                'c': (16, 24, 2),
                's': [1],
                'se': None,
            }),
            ('MBConv', {
                'k': [3, 5],
                'e': (1, ),
                'c': (16, 24, 2),
                's': [1],
                'se': [False],
            }),
            ('MBConv', {
                'k': [3, 5],
                'e': (4, 7),
                'c': (20, 32, 4),
                's': [1],
                'se': [True],
            }),
            ('MBConv', {
                'k': [3, 5],
                'e': (4, 7),
                'c': (24, 48, 4),
                's': [1],
                'se': [True],
            })
        ]
    },
    'recipe': {
        '_type': 'recipe_choice',
        '_value': {
            'lr': (30, 40),
            'optim': ['RMSProp', 'SGD'],
            'ema': [True, False],
            'wd': (7, 21)
        }
    }
}


search_space_v2 = {
    'stage1': {
        '_type': 'layer_choice',
        '_value': [
            ('Conv', {
                'k': [1, 3],
                'e': None,
                'c': (3, 9),
                's': [1],
                'se': None,
            }),
            ('Conv', {
                'k': [3, 5],
                'e': None,
                'c': (6, 18, 2),
                's': [1],
                'se': None,
            })
        ]
    },
    'stage2': {
        '_type': 'layer_choice',
        '_value': [
            ('Pool', {
                'k': [2, 3],
                'class': ['Max', 'Avg']
            })
        ]
    },
    'stage3': {
        '_type': 'layer_choice',
        '_value': [
            ('Conv', {
                'k': [1, 3],
                'e': None,
                'c': (9, 15),
                's': [1],
                'se': None,
            }),
            ('Conv', {
                'k': [3, 5],
                'e': None,
                'c': (12, 30, 2),
                's': [1],
                'se': None,
            })
        ]
    },
    'stage4': {
        '_type': 'layer_choice',
        '_value': [
            ('Pool', {
                'k': [2, 3],
                'class': ['Max', 'Avg']
            })
        ]
    },
    'recipe': {
        '_type': 'recipe_choice',
        '_value': {
            'lr': (1, 20),
            'optim': ['RMSProp', 'SGD'],
            'wd': (7, 21)
        }
    }
}

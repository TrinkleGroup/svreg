class Option:
    """Simple helper class for preparing various settings options"""

    def __init__(self, name, allowedType, limits, suggestedValue, message):
        self.name = name
        self.allowedType = allowedType
        self.limits = limits
        self.suggestedValue = suggestedValue
        self.message = message

    def printInfo(self):
        print(
            '{} ({}, limits = {}):\n\n{} Suggested value/range:{}\n'.format(
                self.name, self.allowedType, self.limits, self.message,
                self.suggestedValue
            )
        )


_valid_options = {
    'seed' : Option(
        'seed', int, None, 42,
        'The seed to pass to the random number generators.'
    ),
    'runType': Option(
        'runType', str, None, ['GA', 'DEBUG'],
        'Used for debugging. Default is `GA`, which starts a symbolic'\
            ' regression run. `DEBUG` runs the debug function set in'\
                ' __main__.py.'
    ),
    'useGPU': Option(
        'useGPU', bool, None, False,
        'True to use GPU acceleration. Requires cupy installation.'
    ),
    'refStruct': Option(
        'refStruct', str, None, 'Ground_state_crystal',
        'The database key corresponding to the name of the structure that'\
            ' will be used as the reference structure for energy'\
                ' difference calculations (i.e. E = E_struct - E_ref).'
    ),
    'databasePath': Option(
        'databasePath', str, None, None,
        'The full path to the HDF5 database file.'
    ),
    'outputPath': Option(
        'outputPath', str, None, './results',
        'The path to the folder for storing any outputs. Defaults to'\
        ' `./results`. Creates folder if it does not exist yet.'
    ),
    'overwrite': Option(
        'overwrite', bool, (False, True), False,
        'True if the files in `outputPath` should be overwritten.'
    ),
    'costFxn': Option(
        'costFxn', str, None, 'MAE',
        'The type of cost function to use for single-valued optimization.'
    ),
    'optimizer': Option(
        'optimizer', str, None, 'CMA',
        'The name of the optimizer object to use for optimizing strings.'
    ),
    'allSums': Option(
        'allSums', bool, (False, True), False,
        'If True, only "+" function nodes will be added. This limits the'\
            'non-linearity of the trees, but allows for significantly lower'\
                'computational costs during optimization.'
    ),
    'numberOfTrees': Option(
        'numberOfTrees', int, None, [10, 100],
        'The number of equation trees for the regressor to generate at'\
            ' each step. Too few, and the regression may fail to converge;'\
                ' too many, and the regressor may be slow.'
    ),
    'tournamentSize': Option(
        'tournamentSize', int, (1,), [10, 50],
        'The number of individuals from the current set of trees to use'\
            ' for tournament selection. Usually 10%-20% of population size'
    ),
    'crossoverProb': Option(
        'crossoverProb', float, (0, 1), 0.2,
        'The probability of performing a crossover operation when evolving'\
            ' trees.'
    ),
    'pointMutateProb': Option(
        'pointMutateProb', float, (0, 1), 0.2,
        'The probability of performing a point mutation when evolving each'\
            ' node in a tree.'
    ),
    'optimizerPopSize': Option(
        'optimizerPopSize', int, None, [10, 100],
        'The number of parameter sets to generate for each tree when '\
            'optimizing the trees. Too few, and the trees may not optimize'\
                ' well; too many, and the regressor may be slow.'
    ),
    'maxTreeDepth': Option(
        'maxTreeDepth', int, (0,), [0, 3],
        'The maximum allowed tree depth. Trees should be kept relatively '\
            'shallow to encourage speed and interpretability of the final '\
                'potential form.'
    ),
    'maxNumSVs': Option(
        'maxNumSVs', int, (1,), 10,
        'The maximum number of structure vectors that a chemistry tree can'\
            'have.'
    ),
    'numRegressorSteps': Option(
        'numRegressorSteps', int, None, [100, 1000],
        'The number of steps for the regressor to take. Should be large '\
            'enough to allow for convergence, and ideally not any larger. '\
                'Realistically, just set it as large as possible given '\
                    'available computational resources, then stop early '\
                        'if regressor begins to converge.'
    ),
    'maxNumOptimizerSteps': Option(
        'maxNumOptimizerSteps', int, None, [100, 1000],
        'The maximum number of steps to take when optimizing tree parameters.'\
            'Should be kept relatively small, but large enough that the tree'\
                'is mostly converged. The final tree should be polished later'\
                    'anyways, so "sloppy" trees are okay during regression.'
    ),
    'energyWeight': Option(
        'energyWeight', float, (0,), 1,
        'The weight of energy errors.'
    ),
    'forcesWeight': Option(
        'forcesWeight', float, (0,), 1,
        'The weight of forces errors.'
    ),
    'ridgePenalty': Option(
        'ridgePenalty', float, (0,), 1,
        'The penalty magnitude for ridge regression.'
    )
}


class Settings(dict):
    """
    Used for managing loading all settings. Simply stores all settings in a
    dictionary, with some checks to ensure valid options are used. Also enables
    loading from input file.

    If a given option isn't specified, then it will be populated with a
    suggested default value.

    Settings files should be formatted as follows:
        <key_name> <value>

    Where <key_name> is the name of the setting, and value is the value. If
    <value> is omitted, then it will automatically populated using a suggested
    value.
    
    Empty lines and lines preceded by '#' are assumed to be comments and are
    ignored.
    """

    @classmethod
    def from_file(cls, filePath):
        settingsStringDict = {}

        with open(filePath, 'r') as settingsFile:
            lines = [line.strip() for line in settingsFile.readlines()]
            for line in lines:
                # skip comments and empty lines
                if (len(line) == 0) or (line[0] == '#'):
                    continue
                else:
                    lineSplit = line.split()
                    key = lineSplit[0]

                    if key not in _valid_options:
                        raise RuntimeError(
                            'Invalid option: {}'.format(key)
                        )

                    option = _valid_options[key]

                    if len(lineSplit) == 2:
                        # Load value if given and convert to correct type
                        value = lineSplit[1]

                        try:
                            if option.allowedType is bool:
                                value = (value == 'True')
                            else:
                                value = option.allowedType(value)
                            settingsStringDict[key] = value
                        except:
                            raise RuntimeError(
                                'Problem casting {} value {} to type {}'.format(
                                    option.name, value, option.allowedType
                                )
                            )

                    elif len(lineSplit) == 1:
                        value = None
                    else:
                        raise RuntimeError(
                            'Too many values provided for {}'.format(key)
                        )

                    # prepared valid option
                    settingsStringDict[key] = value

        return cls.from_dict(settingsStringDict)

    @classmethod
    def from_dict(cls, settingsDict):
        settings = cls()

        for key, val in settingsDict.items():
            if key not in _valid_options:
                raise RuntimeError(
                    'Invalid option: {}'.format(key)
                )

            # Valid option name
            option = _valid_options[key]

            if val is not None:
                # Use provided value

                # Check if invalid type
                if not isinstance(val, option.allowedType):
                    raise RuntimeError(
                        'Invalid type for option "{}". Expected {}'.format(
                            key, option.allowedType
                        )
                    )

                # See if provided value violates any limitations
                limits = option.limits
                if limits is not None:
                    if (val < limits[0]):
                        raise RuntimeError(
                            '{} value must be >= {}'.format(
                                option.name, limits[0]
                            )
                        )
                    elif (len(limits) > 1) and (val > limits[1]):
                        raise RuntimeError(
                            '{} value must be <= {}'.format(
                                option.name, limits[1]
                            )
                        )

            settings[key] = val
        
        # Populate any un-specified options
        for expectedKey in _valid_options:
            if expectedKey not in settings:
                value = _valid_options[key].suggestedValue

                if type(value) == list:
                    # Choose the smallest suggested value if given a range
                    value = value[0]

                settings[expectedKey] = value

        # Final checks
        if settings['tournamentSize'] > settings['numberOfTrees']:
            raise RuntimeError("tournamentSize must be <= numberOfTrees")

        return settings

    
    @classmethod
    def printValidSettings(cls):
        for key, option in _valid_options.items():
            print('-'*20)
            option.printInfo()
        print('-'*20)

    # @staticmethod
    def printSettings(self):
        for key, val in self.items():
            option = _valid_options[key]

            print(
                '{}: {}'.format(
                    key, val
                # '{} (type: {}, value: {}):\n{}\n'.format(
                #     key, option.allowedType, val, option.message
                )
            )


if __name__ == '__main__':
    print("Valid settings:\n")
    Settings.printValidSettings()
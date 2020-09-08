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
    'PROCS_PER_MANAGER': Option(
        'PROCS_PER_MANAGER', int, (1,), 32,
        'The size of the processor farm given to each Manager object;'\
            ' used for tuning parallelization.'
    ),
    'PROCS_PER_PHYS_NODE': Option(
        'PROCS_PER_PHYS_NODE', int, (1,), 32,
        'The number of cores on a physical compute node. The number of'\
            ' compute nodes assigned to each Manager is euqal to'\
                ' (PROCS_PER_MANAGER / PROCS_PER_PHYS_NODE.'
    ),
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
    'optimizer': Option(
        'optimizer', str, None, 'CMA',
        'The name of the optimizer object to use for optimizing strings.'
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
        'maxTreeDepth', int, (1,), [1, 3],
        'The maximum allowed tree depth. Trees should be kept relatively '\
            'shallow to encourage speed and interpretability of the final '\
                'potential form.'
    ),
    'numRegressorSteps': Option(
        'numRegressorSteps', int, None, [100, 1000],
        'The number of steps for the regressor to take. Should be large '\
            'enough to allow for convergence, and ideally not any larger. '\
                'Realistically, just set it as large as possible given '\
                    'available computational resources, then stop early '\
                        'if regressor begins to converge.'
    ),
    'numOptimizerSteps': Option(
        'numOptimizerSteps', int, None, [10, 100],
        'The number of steps to take when optimizing tree parameters.'\
            'Should be kept relatively small. The final tree should be '\
                're-parameterized anyways, so "sloppy" trees are okay '\
                    'during regression.'
    ),
    'energyWeight': Option(
        'energyWeight', float, (0,), 1,
        'The weight of energy errors.'
    ),
    'forcesWeight': Option(
        'forcesWeight', float, (0,), 1,
        'The weight of forces errors.'
    ),
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

    @staticmethod
    def printSettings(self):
        for key, val in self.items():
            option = _valid_options[key]

            print(
                '{} (type: {}, value: {}):\n\n{}\n'.format(
                    key, option.allowedType, val, option.info()
                )
            )


if __name__ == '__main__':
    print("Valid settings:\n")
    Settings.printValidSettings()
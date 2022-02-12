from codecs import register_error
from ctypes import Union
from telnetlib import SE
from tkinter import N
from turtle import down
import numpy as np
import pandas as pd
import sys
import operator


class MacroRandomForest:

    '''
    Open Source implementation of Macroeconomic Random Forest.
    This class runs MRF, where RF is employed to generate generalized time-varying parameters
    for a linear (macroeconomic) equation. See: https://arxiv.org/pdf/2006.12724.pdf for more details.
    '''

    def __init__(self, data, x_pos, oos_pos, y_pos=1,
                 minsize=10, mtry_frac=1/3, min_leaf_frac_of_x=1,
                 VI=False, ERT=False, quantile_rate=None,
                 S_priority_vec=None, random_x=False, trend_push=1, howmany_random_x=1,
                 howmany_keep_best_VI=20, cheap_look_at_GTVPs=True,
                 prior_var=[], prior_mean=[], subsampling_rate=0.75,
                 rw_regul=0.75, keep_forest=False, block_size=12,
                 fast_rw=True, ridge_lambda=0.1, HRW=0,
                 B=50, resampling_opt=2, print_b=True):

        ######## INITIALISE VARIABLES ###########

        # Dataset handling
        self.data, self.x_pos, self.oos_pos, self.y_pos = data, x_pos, oos_pos, y_pos
        self.data.columns = [i for i in range(len(data.columns))]

        # Properties of the tree
        self.minsize, self.mtry_frac, self.min_leaf_frac_of_x = minsize, mtry_frac, min_leaf_frac_of_x

        # [Insert general categorisation]
        self.VI, self.ERT, self.quantile_rate, self.S_priority_vec = VI, ERT, quantile_rate, S_priority_vec

        # [Insert general categorisation]
        self.random_x, self.howmany_random_x, self.howmany_keep_best_VI = random_x, howmany_random_x, howmany_keep_best_VI

        # [Insert general categorisation]
        self.cheap_look_at_GTVPs, self.prior_var, self.prior_mean = cheap_look_at_GTVPs, prior_var, prior_mean

        # [Insert general categorisation]
        self.subsampling_rate, self.rw_regul, self.keep_forest = subsampling_rate, rw_regul, keep_forest

        # [Insert general categorisation]
        self.block_size, self.fast_rw = block_size, fast_rw

        # [Insert general categorisation]
        self.ridge_lambda, self.HRW, self.B, self.resampling_opt, self.print_b = ridge_lambda, HRW, B, resampling_opt, print_b

        self.S_pos = np.arange(1, len(data.columns))
        self.trend_pos = max(self.S_pos)
        self.trend_push = trend_push

        ######################################

        ######## RUN LOGISTICS ###########
        self._name_translations()
        self._array_setup()
        self._input_safety_checks()

        ######## TRAIN FOREST ###########
        self._ensemble_loop()

    def _name_translations(self):
        '''
        Translation block and misc.
        '''

        self.ET = self.ERT  # translation to older names in the code
        self.ET_rate = self.quantile_rate  # translation to older names in the code
        self.z_pos = self.x_pos  # translation to older names in the code
        self.x_pos = self.S_pos  # translation to older names in the code
        self.random_z = self.random_x  # translation to older names in the code
        # translation to older names in the code
        self.min_leaf_fracz = self.min_leaf_frac_of_x
        # translation to older names in the code
        self.random_z_number = self.howmany_random_x
        self.x_pos = self.S_pos  # translation to older names in the code
        self.VI_rep = 10*self.VI  # translation to older names in the code
        # translation to older names in the code
        self.bootstrap_opt = self.resampling_opt
        self.regul_lambda = self.ridge_lambda  # translation to older names in the code
        self.prob_vec = self.S_priority_vec  # translation to older names in the code
        # translation to older names in the code
        self.keep_VI = self.howmany_keep_best_VI
        # used to be an option, but turns out I can't think of a good reason to choose 'False'. So I impose it.
        self.no_rw_trespassing = True
        self.BS4_frac = self.subsampling_rate  # translation to older names in the code

        ################### INTERNAL NOTE: RYAN ###################
        # VERIFY np.where has exact same functionality as which() from R.
        # translate self.trend_pos to be interms of position in S
        self.trend_pos = np.where(self.S_pos == self.trend_pos)
        ################### INTERNAL NOTE: RYAN ###################

        if self.prior_var != []:
            # those are infact implemented in terms of heterogeneous lambdas
            self.prior_var = 1/self.prior_var

    def _input_safety_checks(self):
        '''
        Sanity checks of input variables
        '''

        if len(self.z_pos) > 10 and (self.regul_lambda < 0.25 or self.rw_regul == 0):
            raise Exception(
                'For models of this size, consider using a higher ridge lambda (>0.25) and RW regularization.')
        if(min(self.x_pos) < 1):
            raise Exception('S.pos cannot be non-postive.')
        if(max(self.x_pos) > len(self.data.columns)):
            raise Exception(
                'S.pos specifies variables beyond the limit of your data matrix.')

        ################### INTERNAL NOTE: RYAN ###################
        # Another use of np.where to replace which(). Need to check it.
        if self.y_pos in list(self.x_pos):
            self.x_pos = self.x_pos[self.x_pos != self.y_pos]
            print('Beware: foolproof 1 activated. self.S_pos and self.y_pos overlap.')
            self.trend_pos = np.where(
                self.x_pos == self.trend_pos)  # foolproof 1
        ################### INTERNAL NOTE: RYAN ###################

        if self.y_pos in list(self.z_pos):
            self.z_pos = self.z_pos[self.z_pos != self.y_pos]
            # foolproof 2
            print('Beware: foolproof 2 activated. self.x_pos and self.y_pos overlap.')

        if self.regul_lambda < 0.0001:
            self.regul_lambda = 0.0001
            # foolproof 3
            print('Ridge lambda was too low or negative. Was imposed to be 0.0001')

        if self.ET_rate != None:
            if self.ET_rate < 0:
                self.ET_rate = 0
                # foolproof 4
                print('Quantile Rate/ERT rate was forced to 0, It cannot be negative.')

        if self.prior_mean != None and self.random_z:
            print('Are you sure you want to mix a customized prior with random X?')

        if len(self.z_pos) < 1:
            raise Exception('You need to specificy at least one X.')

        if len(self.prior_var) != 0 or len(self.prior_mean) != 0:
            if self.prior_var == None and self.prior_mean != None:
                raise Exception('Need to additionally specificy prior_var.')
            if self.prior_mean == None and self.prior_var != None:
                raise Exception('Need to additionally specificy prior_mean.')
            if len(self.prior_mean) != len(self.z_pos) + 1 or len(self.prior_var) != len(self.z_pos) + 1:
                raise Exception(
                    'Length of prior vectors do not match that of X.')

        if len(self.x_pos) < 5:
            print(
                'Your S_t is small. Consider augmentating with noisy carbon copies it for better results.')
        if len(self.x_pos) * self.mtry_frac < 1:
            raise Exception('Mtry.frac is too small.')
        if self.min_leaf_fracz > 2:
            self.min_leaf_fracz = 2
            print('min.leaf.frac.of.x forced to 2. Let your trees run deep!')

        # print(self.data.columns)
        # if self.data.columns == None:
        #     raise Exception('Data frame must have column names.')

        if self.min_leaf_fracz*(len(self.z_pos)+1) < 2:
            self.min_leaf_fracz = 2/(len(self.z_pos)+1)
            print(f'Min.leaf.frac.of.x was too low. Thus, it was forced to ', 2 /
                  ({len(self.z_pos)}+1), ' -- a bare minimum. You should consider a higher one.', sep='')

        if len(self.oos_pos) == 0:
            self.oos_flag = True
            self.shit = np.zeros(shape=(2, len(self.data.columns)))
            self.shit.columns = self.data.columns
            self.new_data = pd.concat([self.data, self.shit])
            self.original_data = self.data.copy()
            self.data = self.new_data
            self.oos_pos = [len(self.data)-1, len(self.data)]
            self.fake_pos = self.oos_pos

            ################### INTERNAL NOTE: RYAN ###################
            # Does this have the same effect as rownames = c() in R? I think this may be a genuine difference.
            self.data.index = None
            ################### INTERNAL NOTE: RYAN ###################

        else:
            self.oos_flag = False

    def _array_setup(self):
        '''
        Initialising helpful arrays.
        '''

        self.dat = self.data
        self.K = len(self.z_pos)+1
        if self.random_z:
            self.K = self.random_z_number+1

        self.commitee = np.tile(np.nan, (self.B, len(self.oos_pos)))

        self.avg_pred = [0]*len(self.oos_pos)

        self.pred_kf = np.stack(
            [np.tile(np.nan, (self.B, len(self.data)))]*(len(self.x_pos)+1))

        self.all_fits = np.tile(
            np.nan, (self.B, len(self.data)-len(self.oos_pos)))

        self.avg_beta = np.zeros(shape=(len(self.data), self.K))

        self.whos_in_mat = np.zeros(shape=(len(self.data), self.B))

        self.beta_draws = np.stack(
            [np.zeros(shape=self.avg_beta.shape)]*self.B)

        self.betas_draws_nonOVF = self.beta_draws

        self.betas_shu = np.stack(
            [np.zeros(shape=self.avg_beta.shape)]*(len(self.x_pos)+1))

        self.betas_shu_nonOVF = np.stack(
            [np.zeros(shape=self.avg_beta.shape)]*(len(self.x_pos)+1))

        self.avg_beta_nonOVF = self.avg_beta

        self.forest = []

        self.random_vecs = []

    def _ensemble_loop(self):
        '''
        Core random forest ensemble loop.
        '''

        self.Bs = np.arange(1, self.B + 1)

        # The original basis for this code is taken from publicly available code for a simple tree by André Bleier.
        # Standardize data (remeber, we are doing ridge in the end)

        self.std_stuff = standard(self.data)

        for b in self.Bs:

            self._process_subsampling_selection()

            if self.print_b:
                print(f"Tree {b} out of {self.B}")

            if self.random_z:

                self.z_pos_effective = np.random.choice(a=self.z_pos,
                                                        replace=False,
                                                        size=self.random_z_number)
            else:

                self.z_pos_effective = self.z_pos

            self.rt_output = self._one_MRF_tree()

    def _process_subsampling_selection(self):
        '''
        Processes user choice for subsampling technique.
        '''

        # No bootstap/sub-sampling. Should not be used for looking at GTVPs.
        if self.bootstrap_opt == 0:
            self.chosen_ones_plus = [
                1, len(self.data.iloc[-self.oos_pos, :])]  # TRICKY ###

        elif self.bootstrap_opt == 1:  # Plain sub-sampling
            self.chosen_ones = np.random.choice(a=np.arange(0, len(self.data.iloc[-self.oos_pos, :]) + 1),
                                                replace=False,
                                                size=self.BS4_frac*len(self.data.iloc[-self.oos_pos, :]))  # Is size equivalent to n here? Why both options appear in R?
            self.chosen_ones_plus = list(self.chosen_ones)
            self.rando_vec = list(sorted(self.chosen_ones_plus))

        # Block sub-sampling. Recommended when looking at TVPs.
        elif self.bootstrap_opt == 2:

            self.n_obs = self.data[:self.oos_pos[0] - 1].shape[0]

            self.groups = sorted(np.random.choice(
                list(np.arange(0, int(self.n_obs/self.block_size))), size=self.n_obs, replace=True))

            self.rando_vec = np.random.exponential(
                1, size=int(self.n_obs/self.block_size)) + 0.1

            self.rando_vec = [self.rando_vec[i] for i in self.groups]

            self.chosen_ones_plus = self.rando_vec

            self.rando_vec = np.where(self.chosen_ones_plus > np.quantile(
                self.chosen_ones_plus, 1 - self.BS4_frac))[0]

            self.chosen_ones_plus = self.rando_vec

            self.rando_vec = sorted(self.chosen_ones_plus.tolist())

        elif self.bootstrap_opt == 3:  # Plain bayesian bootstrap
            self.chosen_ones = np.random.exponential(
                scale=1, size=len(self.data.iloc[-self.oos_pos, :]))
            self.chosen_ones_plus = self.chosen_ones/np.mean(self.chosen_ones)
            self.rando_vec = self.chosen_ones_plus

        # Block Bayesian Bootstrap. Recommended for forecasting.
        elif self.bootstrap_opt == 4:

            pass
            ################### INTERNAL NOTE: RYAN ###################
            # To hear from Isaac
            ################### INTERNAL NOTE: RYAN ###################

    def _one_MRF_tree(self):
        '''
        Function to create a single MRF tree.
        '''

        data = self.std_stuff["Y"]

        # Adjust prior.mean according to standarization
        if len(self.prior_mean) != 0:
            self.prior_mean[-1] = (1/self.std_stuff["std"][:, self.y_pos]) * \
                (self.prior_mean[-1]*self.std_stuff["std"][:, self.z_pos])

            self.prior_mean[1] = (self.prior_mean[1]-self.std_stuff["mean"][:, self.y_pos] +
                                  self.std_stuff["mean"][:, self.z_pos]@self.prior_mean[-1])/self.std_stuff["std"][:, self.y_pos]

        # Just a simple/convenient way to detect you're using BB or BBB rather than sub-sampling
        if min(self.rando_vec) < 0:
            self.weights = self.rando_vec
            self.rando_vec = np.arange(0, min(self.oos_pos))
            self.bayes = True

        else:
            self.bayes = False

        if self.minsize < (2 * self.min_leaf_fracz * (len(self.z_pos)+2)):
            self.minsize = 2*self.min_leaf_fracz*(len(self.z_pos)+1)+2

        # coerce to a dataframe
        self.data_ori = pd.DataFrame(data.copy())

        noise = 0.00000015 * \
            np.random.normal(size=len(self.rando_vec))

        data = pd.DataFrame(data)

        data = data.iloc[self.rando_vec,
                         :].add(noise, axis=0)

        self.rw_regul_dat = pd.DataFrame(
            self.data_ori.iloc[: self.oos_pos[0] - 1, [0] + list(self.z_pos)])

        row_of_ones = pd.Series([1]*len(data), index=data.index)

        X = pd.concat([row_of_ones,
                       data.iloc[:, list(self.x_pos)]], axis=1)

        self.y = data.iloc[:, self.y_pos]

        self.z = data.iloc[:, self.z_pos]

        if self.bayes:

            self.z = pd.DataFrame(self.weights*self.z)
            self.y = pd.DataFrame(self.weights*self.y)

        do_splits = True

        tree_info = {"NODE": 1, "NOBS": len(
            data), "FILTER": None, "TERMINAL": "SPLIT"}

        for i in range(1, len(self.z_pos) + 2):
            tree_info[f'b0.{i}'] = 0

        tree_info = pd.DataFrame(tree_info, index=[0])

        while do_splits:

            self.to_calculate = tree_info[tree_info['TERMINAL']
                                          == "SPLIT"].index.tolist()

            all_stop_flags = []

            for j in self.to_calculate:

                # Handle root node
                if tree_info.loc[j, "FILTER"] != None:
                    # subset data according to the filter

                    parsed_filter = tree_info.loc[j,
                                                  "FILTER"].replace("[", "data[")

                    this_data = data[eval(parsed_filter)]

                    column_binded_data = data.copy()

                    column_binded_data.insert(
                        0, "rando_vec", self.rando_vec)

                    self.find_out_who = column_binded_data[eval(
                        parsed_filter.replace("data", "column_binded_data"))]

                    self.whos_who = self.find_out_who.iloc[:, 1]

                    # Get the design matrix

                    X = this_data.iloc[:, self.x_pos]

                    X.insert(0, "Intercept", [1]*len(X))

                    self.y = this_data.iloc[:, self.y_pos]
                    self.z = this_data.iloc[:, self.z_pos]

                    if self.bayes:
                        self.z = pd.DataFrame(
                            self.weights[self.whos_who]*self.z)
                        self.y = pd.DataFrame(
                            self.weights[self.whos_who]*np.matrix(self.y))

                else:
                    this_data = data
                    self.whos_who = self.rando_vec

                    if self.bayes:

                        this_data = self.weights * data

                ####### INTERNAL NOTE: ASK PHILLIPE ABOUT THIS b0 thing ######
                self.old_b0 = tree_info.loc[j, "b0.1"]

                ############## Select potential candidates for this split ###############
                SET = X.iloc[:, 1:]  # all X's but the intercept

                # if(y.pos<trend.pos){trend.pos=trend.pos-1} #so the user can specify trend pos in terms of position in the data matrix, not S_t
                # modulation option

                if self.prob_vec == None:
                    prob_vec = np.repeat(1, repeats=len(SET.columns))

                if self.trend_push > 1:
                    prob_vec[self.trend_pos] = self.trend_push

                ####### INTERNAL NOTE: ASK PHILLIPE ABOUT prob.vec. Currently it looks like [1,1,1,1,..., 1, 4] ######
                prob_vec = np.array([value/sum(prob_vec)
                                     for value in prob_vec])

                ### Does this just mean that column has a 4x prob of selection? In python this doesnt work the same ###

                # classic mtry move
                select_from = np.random.choice(np.arange(0, len(
                    SET.columns)), size=round(len(SET.columns)*self.mtry_frac), p=prob_vec, replace=False)

                if len(SET.columns) < 5:
                    select_from = np.arange(0, len(SET.columns))

                splitting = SET.iloc[:, select_from].apply(
                    lambda x: self._splitter_mrf(x))

                stop_flag = all(splitting.iloc[0, :] == np.inf)

                tmp_splitter = splitting.iloc[0, :].idxmin()

                mn = max(tree_info['NODE'])

                tmp_filter = [f"[{tmp_splitter}] >= {splitting.loc[1, tmp_splitter]}",
                              f"[{tmp_splitter}] < {splitting.loc[1, tmp_splitter]}"]

                ######## INTERNAL NOTE: PUT THIS BACK IN ########

                # split_here  <- !sapply(tmp_filter,
                #     FUN = function(x,y) any(grepl(x, x = y)),
                #     y = tree_info$FILTER)

                ######## INTERNAL NOTE: PUT THIS BACK IN ########

                if tree_info.loc[j, "FILTER"] != None:
                    tmp_filter = ["(" + filterr + ")" + " & " + "(" + tree_info.loc[j, 'FILTER'] + ")"
                                  for filterr in tmp_filter]

                tmp_df = pd.DataFrame(tmp_filter).transpose()

                split_filters = [filterr.replace(
                    "[", "this_data[") for filterr in tmp_filter]

                nobs = [len(this_data[eval(split_filters[i])])
                        for i in range(len(split_filters))]

                tmp_nobs = pd.concat([tmp_df, pd.DataFrame(nobs).transpose()])

                if any(tmp_nobs.iloc[1] <= self.minsize):
                    split_here = np.repeat(False, repeats=2, axis=0)

                split_here = np.repeat(False, repeats=2, axis=0)

                split_here[tmp_nobs.iloc[1] >= self.minsize] = True

                terminal = np.repeat("SPLIT", repeats=2, axis=0)
                terminal[tmp_nobs.iloc[1] < self.minsize] = "LEAF"
                terminal[tmp_nobs.iloc[1] == 0] = "TRASH"

                if not stop_flag:

                    children = {
                        "NODE": [mn+1, mn+2], "NOBS": tmp_nobs.iloc[1], "FILTER": tmp_filter, "TERMINAL": terminal}

                    for i in range(1, len(self.z_pos) + 2):
                        children[f'b0.{i}'] = [
                            splitting.loc[1 + i, tmp_splitter]]*2

                    children = pd.DataFrame(children)

                else:
                    children = None

                tree_info.loc[j, "TERMINAL"] = "PARENT"

                if stop_flag:
                    tree_info.loc[j, "TERMINAL"] = "LEAF"

                tree_info = pd.concat(
                    [tree_info, children]).reset_index(drop=True)

                do_splits = not (all(tree_info['TERMINAL'] != "SPLIT"))

                all_stop_flags.append(stop_flag)

                # Error handling! check if the splitting rule has already been invoked
                # split_here  <- !sapply(tmp_filter,
                #              FUN = function(x,y) any(grepl(x, x = y)),
                #              y = tree_info$FILTER)

                # # append the splitting rules
                # if (!is.na(tree_info[j, "FILTER"])) {
                #     tmp_filter  <- paste(tree_info[j, "FILTER"],
                #                         tmp_filter, sep = " & ")
                # }

                # tmp_nobs =

            if all(all_stop_flags):
                do_splits = False

        self.ori_y = self.data_ori.iloc[:, self.y_pos]

        self.ori_z = pd.concat([pd.Series(
            [1]*len(self.data_ori)), self.data_ori.iloc[:, self.z_pos]], axis=1)

        leafs = tree_info[tree_info["TERMINAL"] == "LEAF"]

        pga = self._pred_given_tree(leafs)

        beta_bank = pga['beta_bank']
        fitted = pga['fitted']

        ###################################################################################
        ###################################################################################
        ########################## Back to original units #################################
        ###################################################################################
        ###################################################################################

        fitted_scaled = fitted

        fitted = fitted * \
            self.std_stuff['std'][self.y_pos] + \
            self.std_stuff['mean'][self.y_pos]
        betas = beta_bank
        betas[:, 0] = beta_bank[:, 0]*self.std_stuff['std'][self.y_pos] + \
            self.std_stuff['mean'][self.y_pos]

        ###### INTERNAL NOTE: CHECK if kk - 2 instead #######
        for kk in range(2, len(betas.columns)):
            betas[:, kk] = beta_bank[:, kk]*self.std_stuff['std'][self.y_pos] / \
                self.std_stuff['std'][self.z_pos[kk-1]]  # kk - 2? Check
            betas[: 0] = betas[:, 0] - betas[:, kk] * \
                self.std_stuff['mean'][self.z_pos[kk-1]]

        ###### INTERNAL NOTE: CHECK if kk - 2 instead #######

        # if self.VI_rep > 0:
        #     whos_in = np.repeat(np.nan, repeats = len(self.x_pos))
        #     for k in range(0, len(self.x_pos)):
        #         whos_in[k] =

        beta_bank_shu = np.stack(
            [np.zeros(shape=beta_bank.shape)]*(len(self.x_pos)+1))
        fitted_shu = np.zeros(shape=(len(fitted), (len(self.x_pos) + 1)))

        return {"tree": tree_info[tree_info['TERMINAL'] == "LEAF"],
                "fit": fitted.iloc[:self.oos_pos[0]],
                "pred": fitted.iloc[self.oos_pos],
                "data": self.ori_data,
                "betas": betas,
                'betas_shu': beta_bank_shu,
                "fitted_shu": fitted_shu}

        # avg_pred = ((b-1)/b)*avg_pred + (1/b)*fitted[self.oos_pos]

    def _splitter_mrf(self, x):

        cons_w = 0.01

        uni_x = np.unique(x)

        splits = sorted(uni_x)

        z = np.insert(np.matrix(self.z), 0, pd.Series(
            [1]*len(self.z), index=self.z.index), axis=1)
        y = np.matrix(self.y)

        sse = np.repeat(np.inf, repeats=len(uni_x), axis=0)
        the_seq = np.arange(0, len(splits))

        if self.rw_regul <= 0:
            self.fast_rw = True

        if self.ET_rate != None:
            if self.ET and len(z) > 2*self.minsize:
                samp = splits[self.min_leaf_fracz*z.shape[1]                              : len(splits) - self.min_leaf_fracz*z.shape[1] + 1]
                splits = np.random.choice(
                    samp, size=max(1, self.ET_rate*len(samp)), replace=False)
                the_seq = np.arange(0, len(splits))

            elif self.ET == False and len(z) > 4*self.minsize:
                samp = splits[self.min_leaf_fracz*z.shape[1]                              : len(splits) - self.min_leaf_fracz*z.shape[1] + 1]

                splits = np.quantile(samp, np.arange(
                    0.01, 1, (1-0.01)/(max(1, self.ET_rate*len(samp)))))

                the_seq = np.arange(0, len(splits))

        reg_mat = np.identity(z.shape[1])*self.regul_lambda
        reg_mat[0, 0] = cons_w*reg_mat[0, 0]

        if len(self.prior_var) > 0:
            reg_mat = np.diag(np.array(self.prior_var))*self.regul_lambda

        if len(self.prior_mean) == 0:

            b0 = np.linalg.solve(np.matmul(z.T, z) + reg_mat, z.T@y.T)

        else:

            # WIP #
            b0 = np.linalg.solve(np.matmul(
                z.T, z) + reg_mat, np.matmul(z.T, y - z @ self.prior_mean)) + self.prior_mean

        nrrd = self.rw_regul_dat.shape[0]
        ncrd = self.rw_regul_dat.shape[1]

        for i in the_seq:

            sp = splits[i]

            id1 = np.where(x < sp)[0]
            id2 = np.where(x >= sp)[0]

            if len(id1) >= self.min_leaf_fracz*z.shape[1] and len(id2) >= self.min_leaf_fracz*z.shape[1]:
                Id = id1
                yy = y.take([Id])

                zz = z[Id, :]
                zz_privy = zz

                if not self.fast_rw:
                    everybody = (self.whos_who[Id] +
                                 1).union(self.whos_who[Id]-1)
                    everybody = [
                        a for a in everybody if not a in self.whos_who]
                    everybody = everybody[everybody > 0]
                    everybody = everybody[everybody < nrrd + 1]

                    if self.no_rw_trespassing:
                        everybody = np.intersect1d(everybody, self.rando_vec)
                    everybody2 = (self.whoswho[Id]+2).union(self.whoswho[Id]-2)
                    everybody2 = [
                        a for a in everybody2 if not a in self.whos_who]
                    everybody2 = [a for a in everybody2 if not a in everybody]
                    everybody2 = everybody2[everybody2 > 0]
                    everybody2 = everybody2[everybody2 < nrrd + 1]

                    if self.no_rw_trespassing:
                        everybody2 = np.intersect1d(everybody2, self.rando_vec)

                    if len(everybody) == 0:
                        y_neighbors = None
                        z_neighbors = None

                    else:
                        y_neighbors = np.matrix(
                            self.rw_regul*self.rw_regul_dat[everybody, 0])
                        z_neighbours = np.matrix(self.rw_regul * np.hstack(
                            np.repeat(1, repeats=len(everybody2)),
                            np.matrix(self.rw_regul_dat[everybody2, 1: ncrd+1])))

                    if len(everybody2) == 0:
                        y_neighbors2 = None
                        z_neighbors2 = None

                    else:
                        y_neighbors2 = np.matrix(
                            self.rw_regul ** 2 * self.rw_regul_dat[everybody2, 0])
                        z_neighbours2 = np.matrix(self.rw_regul ** 2*np.hstack(
                            np.repeat(1, repeats=len(everybody2)),
                            np.matrix(self.rw_regul_dat[everybody2, 1: ncrd+1])))

                    yy = yy.append(y_neighbors).append(y_neighbors2)
                    zz = np.vstack(np.matrix(zz), z_neighbors, z_neighbors2)

                # bvars or not
                if len(self.prior_mean) == 0:

                    # if len(yy) != zz.shape[0]:
                    #     print(f'{len(yy)} and {zz.shape[0]}')

                    p1 = zz_privy@((1-self.HRW)*np.linalg.solve(np.matmul(zz.T, zz) +
                                                                reg_mat, np.matmul(zz.T, yy.T)) + self.HRW*b0)

                else:
                    pass

                    ###### INTERNAL NOTE: RYAN ######
                    # p1 = zz_privy@((1-self.HRW)*np.linalg.solve(np.matmul(zz, zz.T) + reg_mat, np.matmul(
                    # zz, yy - zz @ self.prior_mean)) + self.prior_mean + self.HRW*b0)
                    # WIP

                   ###### INTERNAL NOTE: RYAN ######

                Id = id2
                yy = y.take([Id])
                zz = z[Id, :]
                zz_privy = zz

                if not self.fast_rw:
                    everybody = (
                        self.whos_who[Id]+1).union(self.whos_who[Id]-1)
                    everybody = [
                        a for a in everybody if not a in self.whos_who]
                    everybody = everybody[everybody > 0]
                    everybody = everybody[everybody < nrrd + 1]

                    if self.no_rw_trespassing:
                        everybody = np.intersect1d(everybody, self.rando_vec)
                    everybody2 = (self.whoswho[Id]+2).union(self.whoswho[Id]-2)
                    everybody2 = [
                        a for a in everybody2 if not a in self.whos_who]
                    everybody2 = [a for a in everybody2 if not a in everybody]
                    everybody2 = everybody2[everybody2 > 0]
                    everybody2 = everybody2[everybody2 < nrrd + 1]

                    if self.no_rw_trespassing:
                        everybody2 = np.intersect1d(
                            everybody2, self.rando_vec)

                    if len(everybody) == 0:
                        y_neighbors = None
                        z_neighbors = None

                    else:
                        y_neighbors2 = np.matrix(
                            (self.rw_regul ** 2) * self.rw_regul_dat[everybody2, 1])
                        z_neighbours2 = np.matrix(self.rw_regul ** 2*np.hstack(
                            np.repeat(1, repeats=len(everybody2)),
                            np.matrix(self.rw_regul_dat[everybody2, 1: ncrd])))

                    yy = yy.append(y_neighbors).append(y_neighbors2)
                    zz = np.vstack(np.matrix(zz), z_neighbors,
                                   z_neighbors2, None)

                if len(self.prior_mean) == 0:

                    p2 = zz_privy@((1-self.HRW)*np.linalg.solve(np.matmul(zz.T, zz) +
                                                                reg_mat, np.matmul(zz.T, yy.T)) + self.HRW*b0)

                else:

                    pass
                    ###### INTERNAL NOTE: RYAN ######
                    # p2 = zz_privy@((1-self.HRW)*np.linalg.solve(np.matmul(zz.T, zz) + reg_mat, np.matmul(
                    #    zz, yy - zz @ self.prior_mean)) + self.prior_mean+self.HRW*b0)
                    # WIP

                   ###### INTERNAL NOTE: RYAN ######

                sse[i] = sum(np.subtract(list(y.take([id1]).flat), list(p1.flat)) ** 2) + \
                    sum(np.subtract(
                        list(y.take([id2]).flat), list(p2.flat)) ** 2)

        # implement a mild preference for 'center' splits, allows trees to run deeper
        sse = DV_fun(sse, DV_pref=0.15)

        split_at = splits[sse.argmin()]

        return pd.Series([min(sse)] + [split_at] + list(b0.flat))

    def _pred_given_tree(self, leafs):

        fitted = np.repeat(np.nan, repeats=len(self.ori_y))

        beta_bank = np.full(fill_value=np.nan, shape=(
            len(self.ori_y), len(self.ori_z.columns)))

        for i in range(0, len(leafs)):

            ind_all = list(self.data_ori[eval(
                leafs['FILTER'].iloc[i].replace("[", "self.data_ori["))].index)

            ind = [i for i in ind_all if i < self.oos_pos[0]]

            if len(ind_all) > 0:

                yy = self.ori_y[ind]

                if len(ind) == 1:
                    zz = np.transpose(np.matrix(self.ori_z.iloc[ind, :]))
                    zz_all = np.matrix(self.ori_z.iloc[ind_all, :])

                    if len(ind_all) == 1:
                        zz_all = np.transpose(
                            np.matrix(self.ori_z.iloc[ind_all, :]))

                else:

                    zz = np.matrix(self.ori_z.iloc[ind, :])
                    zz_all = self.ori_z.iloc[ind_all, :]

                # Simple ridge prior

                reg_mat = np.identity(len(self.z_pos) + 1)*self.regul_lambda

                reg_mat[0, 0] = 0.01 * reg_mat[0, 0]

                # Adds RW prior in the mix

                if self.rw_regul > 0:

                    everybody = np.unique(
                        np.append(np.array(ind)+1, np.array(ind)-1))

                    everybody = [i for i in everybody if i not in ind_all]
                    everybody = [i for i in everybody if i > 0]
                    everybody = [i for i in everybody if i <
                                 len(self.rw_regul_dat)]

                    everybody2 = np.unique(
                        np.append(np.array(ind)+2, np.array(ind)-2))
                    everybody2 = [i for i in everybody2 if i not in ind_all]
                    everybody2 = [i for i in everybody2 if i > 0]
                    everybody2 = [i for i in everybody2 if i <
                                  len(self.rw_regul_dat)]
                    everybody2 = [i for i in everybody2 if i not in everybody]

                    if self.no_rw_trespassing:
                        everybody2 = np.intersect1d(
                            everybody, self.rando_vec)

                    if len(everybody) == 0:
                        y_neighbors = None
                        z_neighbors = None
                    else:

                        y_neighbors = np.matrix(
                            self.rw_regul * self.rw_regul_dat.iloc[everybody, 0])

                        z_neighbors = np.matrix(self.rw_regul * np.column_stack([np.repeat(1, repeats=len(everybody)).T, np.matrix(
                            self.rw_regul_dat.iloc[everybody, 1: len(self.rw_regul_dat.columns)])]))

                    if len(everybody2) == 0:
                        y_neighbors2 = None
                        z_neighbors2 = None

                    else:
                        y_neighbors2 = np.matrix(
                            self.rw_regul**2 * self.rw_regul_dat.iloc[everybody2, 0])
                        z_neighbors2 = np.matrix(self.rw_regul**2 * np.column_stack([np.repeat(1, repeats=len(
                            everybody2)).T, np.matrix(self.rw_regul_dat.iloc[everybody2, 1: len(self.rw_regul_dat.columns)])]))

                    yy = np.append(
                        np.append(np.array(yy), y_neighbors), y_neighbors2)

                    if len(zz) == len(self.z_pos) + 1:

                        zz = np.vstack(
                            [np.transpose(zz), z_neighbors, z_neighbors2])

                    else:
                        zz = np.vstack([zz, z_neighbors, z_neighbors2])

                    if len(self.prior_var) != 0:
                        reg_mat = np.diag(
                            np.array(self.prior_var))*self.regul_lambda
                        prior_mean_vec = self.prior_mean

                        beta_hat = np.linalg.solve(np.matmul(
                            zz.T, zz) + reg_mat, np.matmul(zz.T, yy - zz @ prior_mean_vec)) + prior_mean_vec

                        ###### INTERNAL NOTE: CHECK INDEXING HERE ########
                        b0 = np.transpose(
                            np.matrix(leafs.iloc[i, 4: 4+len(self.z_pos)]))
                        ###### INTERNAL NOTE: CHECK INDEXING HERE ########
                    else:

                        beta_hat = np.linalg.solve(
                            np.matmul(zz.T, zz) + reg_mat, np.matmul(zz.T, yy).T)

                        b0 = np.transpose(
                            np.matrix(leafs.iloc[i, 4: 4+len(self.z_pos)]))

                    if len(ind_all) == 1:
                        if np.matrix(np.transpose(zz_all)).shape[0] != 1:
                            zz_all = np.transpose(zz_all)

                        fitted[ind_all] = zz_all @ ((1-self.HRW)
                                                    * beta_hat + self.HRW*b0)
                        beta_bank[ind_all, ] = np.tile(A=np.transpose(
                            (1-self.HRW)*beta_hat+self.HRW*b0), reps=(len(ind_all), 1))

        return {"fitted": fitted, "beta_bank": beta_bank}

    def _pred_given_mrf(self):

        return None


def DV_fun(sse, DV_pref=0.25):
    '''
    implement a middle of the range preference for middle of the range splits.
    '''

    seq = np.arange(0, len(sse))
    down_voting = 0.5*seq**2 - seq
    down_voting = down_voting/np.mean(down_voting)
    down_voting = down_voting - min(down_voting) + 1
    down_voting = down_voting**DV_pref

    return sse*down_voting


def standard(Y):
    '''
    Function to standardise the data. Remember we are doing ridge.
    '''

    Y = np.matrix(Y)

    size = Y.shape

    mean_y = Y.mean(axis=0)
    sd_y = Y.std(axis=0)

    Y0 = (Y - np.repeat(mean_y,
          repeats=size[0], axis=0)) / np.repeat(sd_y, repeats=size[0], axis=0)

    return {"Y": Y0, "mean": mean_y, "std": sd_y}

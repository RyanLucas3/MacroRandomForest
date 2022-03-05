import numpy as np
import pandas as pd
import math
from joblib import Parallel, delayed


class MacroRandomForest:

    '''
    Open Source implementation of Macroeconomic Random Forest.
    This class runs MRF, where RF is employed to generate generalized time-varying parameters
    for a linear (macroeconomic) equation. See: https://arxiv.org/pdf/2006.12724.pdf for more details.
    '''

    def __init__(self, data, x_pos, oos_pos, S_pos='', y_pos=1,
                 minsize=10, mtry_frac=1/3, min_leaf_frac_of_x=1,
                 VI=False, ERT=False, quantile_rate=None,
                 S_priority_vec=None, random_x=False, trend_push=1, howmany_random_x=1,
                 howmany_keep_best_VI=20, cheap_look_at_GTVPs=True,
                 prior_var=[], prior_mean=[], subsampling_rate=0.75,
                 rw_regul=0.75, keep_forest=False, block_size=12,
                 fast_rw=True, ridge_lambda=0.1, HRW=0,
                 B=50, resampling_opt=2, print_b=True, parallelise=True, n_cores=-1):

        ######## INITIALISE VARIABLES ###########

        # Dataset handling
        self.data, self.x_pos, self.oos_pos, self.y_pos, self.S_pos = data, x_pos, oos_pos, y_pos, S_pos

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
        self.block_size, self.fast_rw, self.parallelise, self.n_cores = block_size, fast_rw, parallelise, n_cores

        # [Insert general categorisation]
        self.ridge_lambda, self.HRW, self.B, self.resampling_opt, self.print_b = ridge_lambda, HRW, B, resampling_opt, print_b

        if isinstance(self.S_pos, str):
            self.S_pos = np.arange(1, len(self.data.columns))

        self.trend_pos = max(self.S_pos)
        self.trend_push = trend_push
        self.cons_w = 0.01

        if len(self.prior_mean) != 0:
            self.have_prior_mean = True
        else:
            self.have_prior_mean = False

        ######## RUN LOGISTICS ###########
        self._name_translations()
        self._array_setup()
        self._input_safety_checks()

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
        self.trend_pos = np.where(self.S_pos == self.trend_pos)

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

        if self.y_pos in list(self.x_pos):
            self.x_pos = self.x_pos[self.x_pos != self.y_pos]
            print('Beware: foolproof 1 activated. self.S_pos and self.y_pos overlap.')
            self.trend_pos = np.where(
                self.x_pos == self.trend_pos)  # foolproof 1

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

        if len(self.prior_var) != 0 or self.have_prior_mean:
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

        Bs = np.arange(0, self.B)

        # The original basis for this code is taken from publicly available code for a simple tree by André Bleier.
        # Standardize data (remeber, we are doing ridge in the end)

        self.std_stuff = standard(self.data)

        if self.parallelise:

            result = Parallel(n_jobs=self.n_cores)(delayed(self._one_MRF_tree)(b)
                                                   for b in Bs)

        else:
            
            result = [self._one_MRF_tree(b) for b in Bs]

        for b in Bs:

            rt_output = result[b]

            if self.random_z:

                z_pos_effective = np.random.choice(a=self.z_pos,
                                                   replace=False,
                                                   size=self.random_z_number)
            else:

                z_pos_effective = self.z_pos

            rando_vec = rt_output["rando_vec"]
            if self.keep_forest:
                self.forest[[b]] = rt_output['tree']
                self.random_vec[[b]] = rando_vec

            # if(bootstrap.opt==3 |bootstrap.opt==4){ #for Bayesian Bootstrap, gotta impose a cutoff on what is OOS and what is not.
            #     rando.vec=which(chosen.ones.plus>quantile(chosen.ones.plus,.5))
            #     rt.output$betas[is.na(rt.output$betas)]=0
            #     rt.output$pred[is.na(rt.output$pred)]=0
            #     }

            self.commitee[b, :] = rt_output['pred']
            self.avg_pred = pd.DataFrame(self.commitee).mean(axis=0)

            in_out = np.repeat(0, repeats=len(self.data))

            for i in range(len(self.data)):
                if i not in rando_vec:
                    in_out[i] = 1

            self.whos_in_mat[:, b] = in_out

            if b == 0:
                b_avg = b+1

            self.avg_beta = ((b_avg-1)/b_avg)*np.array(self.avg_pred) + \
                (1/b_avg)*rt_output['pred']

            self.beta_draws[b, :, :] = rt_output['betas']

            rt_output['betas'][np.where(in_out == 0), :] = np.repeat(
                0, repeats=len(z_pos_effective) + 1)

            self.avg_beta_nonOVF = self.avg_beta_nonOVF + \
                rt_output['betas']

            rt_output['betas'][np.where(in_out == 0), :] = np.repeat(
                np.nan, repeats=len(z_pos_effective) + 1)

            self.betas_draws_nonOVF[b] = rt_output['betas']

            if self.VI_rep > 0:
                self.pred_kf[:, self.b, -rando_vec]
                self.betas_shu = ((b_avg-1)/b_avg)*self.betas_shu + \
                    (1/b_avg)*rt_output['betas_shu']
                rt_output['betas_shu'][:, np.where(in_out == 0), :]
                self.betas_shu_nonOVF = self.betas_shu_nonOVF + \
                    rt_output['betas_shu']

        how_many_in = pd.DataFrame(self.whos_in_mat).sum(axis=1)

        self.avg_beta_nonOVF = self.avg_beta_nonOVF / \
            np.transpose(np.tile(how_many_in, reps=(
                len(z_pos_effective)+1, 1)))

        for kk in range(1, self.betas_shu_nonOVF.shape[0]):
            self.betas_shu_nonOVF[kk, :, :] = self.betas_shu_nonOVF[kk, :, :] / np.transpose(np.tile(how_many_in, reps=(
                len(z_pos_effective)+1, 1)))

        ###################################################################################
        ###################################################################################
        ########################## VI #####################################################
        ###################################################################################
        ###################################################################################

        # if self.VI_rep > 0:
        #     whos_in = np.repeat(
        #         np.nan, repeats=len(self.x_pos))

        #     beta_bank_shu = np.zeros(())

        ###################################################################################
        ###################################################################################

        if self.oos_flag:
            self.avg_beta_nonOVF = self.avg_beta_nonOVF[-self.fake_pos, :]
            self.avg_beta = self.avg_beta[-self.fake_pos, :]
            self.betas_draws_nonOVF = self.betas_draws_nonOVF[-self.fake_pos, :]
            self.betas_draws = self.betas_draws[-self.fake_pos, :]

            # cancel VI_POOS
            self.VI_poos = None

            # cancel pred and the commitee
            self.avg_pred = None
            self.commitee = None

            self.data = self.data[-self.fake_pos, :]

        if self.random_z:
            self.VI_betas = None
            self.VI_betas_nonOVF = None
            self.VI_oob = None
            self.VI_poos = None
            self.impZ = None
            self.avg_beta = None
            self.avg_beta_nonOVF = None
            self.betas_draws = None
            self.betas_draws_nonOVF = None

        # else:

        return {"YandX": self.data.iloc[:, [self.y_pos] + self.z_pos],
                "pred_ensemble": self.commitee,
                "pred": self.avg_pred,
                # "important_S": self.impZ,
                "S_names": self.data.iloc[:, self.x_pos].columns,
                "betas": self.avg_beta_nonOVF,
                "betas_draws_raw": self.beta_draws,
                "betas_draws": self.betas_draws_nonOVF,
                # "VI_betas": self.VI_betas_nonOVF,
                # "VI_oob": self.VI_oob,
                # "VI_oos": self.VI_poos,
                # "VI_betas_raw": self.VI_betas,
                "model": {"forest": self.forest,
                          "data": self.data,
                          "regul_lambda": self.regul_lambda,
                          "prior_var": self.prior_var,
                          "prior_mean": self.prior_mean,
                          "rw_regul": self.rw_regul,
                          "HRW": self.HRW,
                          "no_rw_trespassing": self.no_rw_trespassing,
                          "B": self.B,
                          "random_vecs": self.random_vecs,
                          "y_pos": self.y_pos,
                          "S_pos": self.x_pos,
                          "x_pos": self.z_pos
                          }
                }

    def _process_subsampling_selection(self):
        '''
        Processes user choice for subsampling technique.
        '''

        # No bootstap/sub-sampling. Should not be used for looking at GTVPs.
        if self.bootstrap_opt == 0:

            rando_vec = np.arange(
                0, len(self.data.iloc[:self.oos_pos[0]]))

        elif self.bootstrap_opt == 1:  # Plain sub-sampling
            chosen_ones = np.random.choice(a=np.arange(0, len(self.data.iloc[-self.oos_pos, :]) + 1),
                                           replace=False,
                                           size=self.BS4_frac*len(self.data.iloc[-self.oos_pos, :]))  # Is size equivalent to n here? Why both options appear in R?
            chosen_ones_plus = list(chosen_ones)

            rando_vec = list(sorted(chosen_ones_plus))

        # Block sub-sampling. Recommended when looking at TVPs.
        elif self.bootstrap_opt == 2:

            n_obs = self.data[:self.oos_pos[0] - 1].shape[0]

            groups = sorted(np.random.choice(
                list(np.arange(0, int(n_obs/self.block_size))), size=n_obs, replace=True))

            rando_vec = np.random.exponential(
                1, size=int(n_obs/self.block_size)) + 0.1

            rando_vec = [rando_vec[i] for i in groups]

            chosen_ones_plus = rando_vec

            rando_vec = np.where(chosen_ones_plus > np.quantile(
                chosen_ones_plus, 1 - self.BS4_frac))[0]

            chosen_ones_plus = rando_vec

            rando_vec = sorted(chosen_ones_plus.tolist())

            # self.rando_vec = np.arange(49, 150)

        elif self.bootstrap_opt == 3:  # Plain bayesian bootstrap
            chosen_ones = np.random.exponential(
                scale=1, size=len(self.data.iloc[-self.oos_pos, :]))
            chosen_ones_plus = chosen_ones/np.mean(chosen_ones)
            rando_vec = chosen_ones_plus

        # Block Bayesian Bootstrap. Recommended for forecasting.
        elif self.bootstrap_opt == 4:

            pass
            ################### INTERNAL NOTE: RYAN ###################
            # To hear from Isaac
            ################### INTERNAL NOTE: RYAN ###################

        return rando_vec

    def _one_MRF_tree(self, b):
        '''
        Function to create a single MRF tree.
        '''

        if self.print_b:
            print(f"Tree {b+1} out of {self.B}")

        rando_vec = self._process_subsampling_selection()

        data = self.std_stuff["Y"]

        # Adjust prior.mean according to standarization
        if self.have_prior_mean:
            self.prior_mean[-1] = (1/self.std_stuff["std"][:, self.y_pos]) * \
                (self.prior_mean[-1]*self.std_stuff["std"][:, self.z_pos])

            self.prior_mean[1] = (self.prior_mean[1]-self.std_stuff["mean"][:, self.y_pos] +
                                  self.std_stuff["mean"][:, self.z_pos]@self.prior_mean[-1])/self.std_stuff["std"][:, self.y_pos]

        # Just a simple/convenient way to detect you're using BB or BBB rather than sub-sampling
        if min(rando_vec) < 0:
            weights = rando_vec
            rando_vec = np.arange(0, min(self.oos_pos))
            bayes = True

        else:
            bayes = False

        if self.minsize < (2 * self.min_leaf_fracz * (len(self.z_pos)+2)):
            self.minsize = 2*self.min_leaf_fracz*(len(self.z_pos)+1)+2

        noise = 0.00000015 * \
            np.random.normal(size=len(rando_vec))

        # coerce to a dataframe
        self.data_ori = pd.DataFrame(data.copy())

        data = pd.DataFrame(data)

        data = data.iloc[rando_vec,
                         :].add(noise, axis=0)

        rw_regul_dat = self.data_ori.iloc[: self.oos_pos[0], [
            0] + list(self.z_pos)]

        row_of_ones = pd.Series([1]*len(data), index=data.index)

        X = pd.concat([row_of_ones,
                       data.iloc[:, list(self.x_pos)]], axis=1)

        y = data.iloc[:, self.y_pos]

        z = data.iloc[:, self.z_pos]

        if bayes:

            z = pd.DataFrame(weights*z)
            y = pd.DataFrame(weights*y)

        do_splits = True

        tree_info = {"NODE": 1, "NOBS": len(
            data), "FILTER": None, "TERMINAL": "SPLIT"}

        for i in range(1, len(self.z_pos) + 2):
            tree_info[f'b0.{i}'] = 0

        tree_info = pd.DataFrame(tree_info, index=[0])

        column_binded_data = data.copy()
        column_binded_data.insert(
            0, "rando_vec", rando_vec)

        while do_splits:

            to_calculate = tree_info[tree_info['TERMINAL']
                                     == "SPLIT"].index.tolist()

            all_stop_flags = []

            for j in to_calculate:

                self.wantPrint = False

                # Handle root node

                filterr = tree_info.loc[j, "FILTER"]

                if filterr != None:
                    # subset data according to the filter

                    parsed_filter = filterr.replace("[", "data[")

                    this_data = data[eval(parsed_filter)]

                    find_out_who = column_binded_data.loc[this_data.index]

                    whos_who = find_out_who.iloc[:, 1]

                    # Get the design matrix
                    X = this_data.iloc[:, self.x_pos]
                    X.insert(0, "Intercept", [1]*len(X))

                    y = this_data.iloc[:, self.y_pos]
                    z = this_data.iloc[:, self.z_pos]

                    if bayes:
                        z = weights[whos_who]*z
                        y = weights[whos_who]*np.matrix(y)

                else:

                    this_data = data
                    whos_who = rando_vec

                    if bayes:

                        this_data = weights * data

                self.old_b0 = tree_info.loc[j, "b0.1"]

                ############## Select potential candidates for this split ###############
                SET = X.iloc[:, 1:]  # all X's but the intercept

                # if(self.y_pos<self.trend_pos):
                #     trend_pos=trend_pos-1 #so the user can specify trend pos in terms of position in the data matrix, not S_t

                n_cols_split = len(SET.columns)

                if self.prob_vec == None:
                    prob_vec = np.repeat(1, repeats=n_cols_split)

                if self.trend_push > 1:
                    prob_vec[self.trend_pos] = self.trend_push

                prob_vec = np.array([value/sum(prob_vec)
                                     for value in prob_vec])

                # classic mtry move
                select_from = np.random.choice(np.arange(0, len(
                    SET.columns)), size=round(n_cols_split*self.mtry_frac), p=prob_vec, replace=False)

                if len(SET.columns) < 5:
                    select_from = np.arange(0, n_cols_split)

                splitting = SET.iloc[:, select_from].apply(
                    lambda x: self._splitter_mrf(x, y, z, whos_who, rando_vec, rw_regul_dat))

                splitting = splitting.reset_index(drop=True)

                sse = splitting.iloc[0, :]

                stop_flag = all(np.array(sse) == np.inf)

                tmp_splitter = sse.idxmin()

                mn = max(tree_info['NODE'])

                criteria = round(splitting.loc[1, tmp_splitter], 15)

                tmp_filter = [f"[{tmp_splitter}] >= {criteria}",
                              f"[{tmp_splitter}] < {criteria}"]

                # if tmp_splitter == 7:
                #     print(tmp_filter)

                if filterr != None:
                    tmp_filter = ["(" + filterr + ")" + " & " + "(" + f + ")"
                                  for f in tmp_filter]

                nobs = np.array([splitting.loc[7, tmp_splitter],
                                splitting.loc[6, tmp_splitter]])

                if any(nobs <= self.minsize):
                    split_here = np.repeat(False, repeats=2, axis=0)

                split_here = np.repeat(False, repeats=2, axis=0)
                split_here[nobs >= self.minsize] = True

                terminal = np.repeat("SPLIT", repeats=2, axis=0)
                terminal[nobs < self.minsize] = "LEAF"
                terminal[nobs == 0] = "TRASH"

                if not stop_flag:

                    children = {"NODE": [
                        mn+1, mn+2], "NOBS": nobs, "FILTER": tmp_filter, "TERMINAL": terminal}

                    for i in range(1, len(self.z_pos) + 2):
                        children[f'b0.{i}'] = [
                            splitting.loc[1 + i, tmp_splitter]]*2

                    children = pd.DataFrame(children)

                tree_info.loc[j, "TERMINAL"] = "PARENT"

                if stop_flag:
                    tree_info.loc[j, "TERMINAL"] = "LEAF"

                tree_info = pd.concat(
                    [tree_info, children]).reset_index(drop=True)

                do_splits = not (
                    all(np.array(tree_info['TERMINAL']) != "SPLIT"))

                all_stop_flags.append(stop_flag)

            if all(all_stop_flags):
                do_splits = False

        self.ori_y = self.data_ori.iloc[:, self.y_pos]

        self.ori_z = pd.concat([pd.Series(
            [1]*len(self.data_ori)), self.data_ori.iloc[:, self.z_pos]], axis=1)

        leafs = tree_info[tree_info["TERMINAL"] == "LEAF"]

        pga = self._pred_given_tree(leafs, rando_vec, rw_regul_dat)

        beta_bank = pga['beta_bank']

        fitted = pga['fitted']

        ###################################################################################
        ###################################################################################
        ########################## Back to original units #################################
        ###################################################################################
        ###################################################################################

        fitted_scaled = fitted

        fitted = fitted * \
            self.std_stuff['std'].flat[self.y_pos] + \
            self.std_stuff['mean'].flat[self.y_pos]

        betas = beta_bank

        betas[:, 0] = beta_bank[:, 0]*self.std_stuff['std'].flat[self.y_pos] + \
            self.std_stuff['mean'].flat[self.y_pos]

        ###### INTERNAL NOTE: CHECK if kk - 2 instead #######
        for kk in range(1, betas.shape[1]):

            betas[:, kk] = beta_bank[:, kk]*self.std_stuff['std'].flat[self.y_pos] / \
                self.std_stuff['std'].flat[self.z_pos[kk-1]]  # kk - 2? Check

            betas[:, 0] = betas[:, 0] - betas[:, kk] * \
                self.std_stuff['mean'].flat[self.z_pos[kk-1]]

        ###### INTERNAL NOTE: CHECK if kk - 2 instead #######

        ####################### VI #######################

        # VI = self._variable_importance # do something with this
        ####################### VI #######################

        beta_bank_shu = np.stack(
            [np.zeros(shape=beta_bank.shape)]*(len(self.x_pos)+1))

        fitted_shu = np.zeros(shape=(len(fitted), (len(self.x_pos) + 1)))

        return {"tree": tree_info[tree_info['TERMINAL'] == "LEAF"],
                "fit": fitted[:self.oos_pos[0]],
                "pred": fitted[self.oos_pos],
                "data": self.data_ori,
                "betas": betas,
                'betas_shu': beta_bank_shu,
                "fitted_shu": fitted_shu,
                "rando_vec": rando_vec}

    def _splitter_mrf(self, x, y, z, whos_who, rando_vec, rw_regul_dat):

        x = np.array(x)

        uni_x = np.unique(x)

        splits = sorted(uni_x)

        # rounded_x = np.round(x.copy(), 7)

        z = np.column_stack([np.ones(len(z)), z])
        min_frac_times_no_cols = self.min_leaf_fracz*z.shape[1]

        y_as_list = np.array(y)
        y = np.matrix(y)

        sse = np.repeat(np.inf, repeats=len(uni_x), axis=0)
        the_seq = np.arange(0, len(splits))

        whos_who = np.array(whos_who)

        nobs1 = {}
        nobs2 = {}

        if self.rw_regul <= 0:
            self.fast_rw = True

        if self.ET_rate != None:
            if self.ET and len(z) > 2*self.minsize:

                samp = splits[min_frac_times_no_cols-1: len(
                    splits) - min_frac_times_no_cols]
                splits = np.random.choice(
                    samp, size=max(1, self.ET_rate*len(samp)), replace=False)
                the_seq = np.arange(0, len(splits))

            elif self.ET == False and len(z) > 4*self.minsize:

                samp = splits[min_frac_times_no_cols-1: len(
                    splits) - min_frac_times_no_cols]

                splits = np.quantile(samp, np.linspace(
                    start=0.01, stop=0.99, num=math.floor(max(1, self.ET_rate*len(samp)))))

                the_seq = np.arange(0, len(splits))

        reg_mat = np.identity(z.shape[1])*self.regul_lambda
        reg_mat[0, 0] = self.cons_w*reg_mat[0, 0]

        if len(self.prior_var) > 0:

            reg_mat = np.diag(np.array(self.prior_var))*self.regul_lambda

        if not self.have_prior_mean:

            z_T = z.T
            b0 = np.linalg.solve(z_T @ z + reg_mat, z_T@y.T)

        else:

            # WIP #
            z_T = z.T
            b0 = np.linalg.solve(
                z_T @ z + reg_mat, np.matmul(z_T, y - z @ self.prior_mean)) + self.prior_mean

        dimension = rw_regul_dat.shape
        nrrd = dimension[0]
        ncrd = dimension[1]

        for i in the_seq:

            sp = splits[i]

            id1 = np.where(x < sp)[0]

            id2 = np.where(x >= sp)[0]

            num_first_split = len(id1)
            num_second_split = len(id2)

            nobs1[i] = num_first_split
            nobs2[i] = num_second_split

            if num_first_split >= min_frac_times_no_cols and num_second_split >= min_frac_times_no_cols:

                yy = np.array([[y_as_list[i] for i in id1]])
                zz = z[id1, :]

                zz_privy = zz

                if not self.fast_rw:

                    everybody = self._find_n_neighbours(
                        whos_who, id1, 1, nrrd, rando_vec)

                    everybody2 = self._find_n_neighbours(
                        whos_who, id1, 2, nrrd, rando_vec)

                    everybody2 = np.array(
                        [a for a in everybody2 if not a in everybody])

                    yy, zz = self._random_walk_regularisation(yy,
                                                              zz,
                                                              everybody,
                                                              everybody2,
                                                              rw_regul_dat,
                                                              ncrd)

                zz_T = zz.T
                # bvars or not
                if not self.have_prior_mean:

                    p1 = zz_privy@((1-self.HRW)*np.linalg.solve(zz_T @ zz +
                                                                reg_mat, zz_T @ yy.T) + self.HRW*b0)

                else:

                    p1 = zz_privy@((1-self.HRW)*np.linalg.solve(zz @ zz_T + reg_mat, np.matmul(
                        zz, yy - zz @ self.prior_mean)) + self.prior_mean + self.HRW*b0)

                yy = np.array([[y_as_list[i] for i in id2]])
                zz = z[id2, :]
                zz_privy = zz

                if not self.fast_rw:

                    everybody = self._find_n_neighbours(
                        whos_who, id2, 1, nrrd, rando_vec)

                    everybody2 = self._find_n_neighbours(
                        whos_who, id2, 2, nrrd, rando_vec)

                    everybody2 = np.array(
                        [a for a in everybody2 if not a in everybody])

                    yy, zz = self._random_walk_regularisation(yy,
                                                              zz,
                                                              everybody,
                                                              everybody2,
                                                              rw_regul_dat,
                                                              ncrd)

                zz_T = zz.T

                if not self.have_prior_mean:

                    p2 = zz_privy@((1-self.HRW)*np.linalg.solve(zz_T @ zz +
                                                                reg_mat, zz_T @ yy.T) + self.HRW*b0)

                else:

                    p2 = zz_privy@((1-self.HRW)*np.linalg.solve(zz_T @ zz + reg_mat, np.matmul(
                        zz, yy - zz @ self.prior_mean)) + self.prior_mean+self.HRW*b0)

                sse[i] = sum(np.subtract(y_as_list.take(id1), np.array(p1.flat)) ** 2) + \
                    sum(np.subtract(
                        y_as_list.take(id2), np.array(p2.flat)) ** 2)

        # implement a mild preference for 'center' splits, allows trees to run deeper
        sse = DV_fun(sse, DV_pref=0.15)

        split_no_chosen = sse.argmin()

        return [min(sse)] + [splits[split_no_chosen]] + list(b0.flat) + [nobs1[split_no_chosen]] + [nobs2[split_no_chosen]]

    def _pred_given_tree(self, leafs, rando_vec, rw_regul_dat):

        fitted = np.repeat(np.nan, repeats=len(self.ori_y))

        beta_bank = np.full(fill_value=np.nan, shape=(
            len(self.ori_y), len(self.ori_z.columns)))

        ori_z = np.matrix(self.ori_z)

        regul_mat = np.matrix(rw_regul_dat)

        leafs_mat = np.matrix(leafs)

        for i in range(0, len(leafs)):

            ind_all = list(self.data_ori[eval(
                leafs_mat[i, 2].replace("[", "self.data_ori["))].index)

            ind = np.array([j for j in ind_all if j <
                            self.oos_pos[0] if not np.isnan(j)])

            if len(ind_all) > 0:

                yy = np.matrix(self.ori_y.iloc[ind])

                if len(ind) == 1:
                    zz = ori_z[ind, :].T
                    zz_all = ori_z[ind_all, :]

                    if len(ind_all) == 1:
                        zz_all = ori_z[ind_all, :].T

                else:
                    zz = ori_z[ind, :]
                    zz_all = ori_z[ind_all, :]

                # Simple ridge prior
                reg_mat = np.identity(len(self.z_pos) + 1)*self.regul_lambda
                reg_mat[0, 0] = 0.01 * reg_mat[0, 0]

                # Adds RW prior in the mix
                nrrd = regul_mat.shape[0]
                ncrd = regul_mat.shape[1]

                if self.rw_regul > 0:

                    everybody = self._find_n_neighbours(
                        ind, None, 1, nrrd, rando_vec, ind_all, "prediction")

                    everybody2 = self._find_n_neighbours(
                        ind, None, 2, nrrd, rando_vec, ind_all, "prediction")

                    everybody2 = np.array(
                        [j for j in everybody2 if j not in everybody])

                    yy, zz = self._random_walk_regularisation(
                        yy, zz, everybody, everybody2, regul_mat, ncrd)

                if len(self.prior_var) != 0:

                    reg_mat = np.diag(
                        np.array(self.prior_var))*self.regul_lambda
                    prior_mean_vec = self.prior_mean

                    zz_T = zz.T

                    beta_hat = np.linalg.solve(
                        zz_T @ zz + reg_mat, np.matmul(zz_T, yy - zz @ prior_mean_vec)) + prior_mean_vec

                    b0 = np.transpose(
                        leafs_mat[i, 4: 4+len(self.z_pos)+1])

                else:

                    zz_T = zz.T

                    beta_hat = np.linalg.solve(
                        np.matmul(zz_T, zz) + reg_mat, np.matmul(zz_T, yy).T)

                    b0 = np.transpose(leafs_mat[i, 4: 4+len(self.z_pos)+1])

                if len(ind_all) == 1:
                    if np.matrix(np.transpose(zz_all)).shape[0] != 1:
                        zz_all = np.transpose(zz_all)

                    fitted_vals = zz_all.T @ ((1-self.HRW)
                                              * beta_hat + self.HRW*b0)

                    for j in range(len(fitted_vals)):
                        fitted[ind_all[j]] = fitted_vals[j]

                else:

                    if zz_all.shape[1] != len(b0):
                        zz_all = zz_all.T

                    fitted_vals = zz_all@((1-self.HRW)
                                          * beta_hat+self.HRW*b0)

                    for j in range(len(fitted_vals)):

                        fitted[ind_all[j]] = fitted_vals[j]

                beta_bank[ind_all, :] = np.tile(A=np.transpose(
                    (1-self.HRW)*beta_hat+self.HRW*b0), reps=(len(ind_all), 1))

        return {"fitted": fitted, "beta_bank": beta_bank}

    def _find_n_neighbours(self, array, i_d, n_neighbours, nrrd, rando_vec, ind_all=None, stage='splitting'):

        if stage == 'splitting':
            everybody_n = np.unique(np.concatenate((array.take(
                i_d)+n_neighbours, array.take(i_d)-n_neighbours)))
            everybody_n = np.array([
                a for a in everybody_n if not a in array])

        elif stage == 'prediction':
            everybody_n = np.unique(np.concatenate(
                (array+n_neighbours, array-n_neighbours)))
            everybody_n = np.array([
                a for a in everybody_n if not a in ind_all])

        # Check that it's a valid index i.e. not -1 (which would incorrectly index the last obsv)
        everybody_n = everybody_n[everybody_n >= 0]
        # check that we're not leaking into OOS
        everybody_n = everybody_n[everybody_n < nrrd]

        if self.no_rw_trespassing:
            everybody_n = np.intersect1d(everybody_n, rando_vec)

        return everybody_n

    def _random_walk_regularisation(self, yy, zz, everybody, everybody2, rw_regul_dat, ncrd):

        add_neighbors = True
        add_neighbors_2 = True

        if len(everybody) == 0:
            add_neighbors = False

        else:
            y_neighbors = np.matrix(
                self.rw_regul*rw_regul_dat[everybody, 0])
            z_neighbors = np.matrix(self.rw_regul * np.column_stack(
                [np.repeat(1, repeats=len(everybody)), rw_regul_dat[everybody, 1: ncrd]]))

        if len(everybody2) == 0:
            add_neighbors_2 = False

        else:
            y_neighbors2 = np.matrix(
                self.rw_regul ** 2 * rw_regul_dat[everybody2, 0])
            z_neighbors2 = np.matrix(self.rw_regul ** 2*np.column_stack(
                [np.repeat(1, repeats=len(everybody2)),
                 np.matrix(rw_regul_dat[everybody2, 1: ncrd])]))

        if len(zz) == len(self.z_pos) + 1:
            zz_copy = np.transpose(zz)

        else:
            zz_copy = zz

        if add_neighbors and add_neighbors_2:

            yy = np.append(
                np.append(np.array(yy), y_neighbors), y_neighbors2)
            zz = np.vstack([zz_copy, z_neighbors, z_neighbors2])

        elif add_neighbors == True and add_neighbors_2 == False:

            yy = np.append(np.array(yy), y_neighbors)
            zz = np.vstack(
                [zz_copy, z_neighbors])

        elif add_neighbors == False and add_neighbors_2 == True:

            yy = np.append(np.array(yy), y_neighbors2)
            zz = np.vstack(
                [zz_copy, z_neighbors2])

        return yy, zz

    def _variable_importance(self, leafs, fitted, fitted_scaled, y, z, rando_vec, rw_regul_dat):

        if self.VI_rep > 0:

            whos_in = np.repeat(np.nan, repeats=len(self.x_pos), axis=0)

            for k in range(len(self.x_pos)):

                x = f"[{self.x_pos[k]}]"

                x_in_filters = False

                for filtr in leafs['FILTER']:
                    if x in filtr:
                        x_in_filters = True

                whos_in[k] = x_in_filters

            beta_bank_shu = np.stack(
                [np.zeros(shape=self.beta_bank.shape)]*(len(self.x_pos)+1))

            fitted_shu = [np.zeros(shape=(len(fitted), len(self.x_pos)+1))]

            for k in range(len(self.x_pos)):
                if whos_in[k]:
                    for ii in range(self.VI_rep):
                        data_shu = self.data_ori
                        data_shu.iloc[:, self.x_pos[k]] = np.random.choice(a=self.data_ori[:, self.x_pos[k]],
                                                                           replace=False,
                                                                           size=len(self.data_ori))

                        pga = self._pred_given_tree(
                            leafs, rando_vec, rw_regul_dat)

                        beta_bank_shu[k+1] = ((ii-1)/ii) * \
                            beta_bank_shu[k+1]+pga['beta_bank']/ii

                        fitted_shu[:, k+1] = ((ii-1)/ii) * \
                            fitted_shu[:, k+1] + pga['fitted']/ii

                else:
                    beta_bank_shu[k+1] = self.beta_bank
                    fitted_shu[:, k+1] = fitted_scaled

        else:
            beta_bank_shu = np.stack(
                [np.zeros(shape=self.beta_bank.shape)]*(len(self.x_pos)+1))
            fitted_shu = [np.zeros(shape=(len(fitted), len(self.x_pos)+1))]

        return beta_bank_shu, fitted_shu

    def financial_evaluation(self, close_prices):

        '''
        Method for generating signals and backtesting the financial performance of MRF
        '''

        daily_profit = []

        T_profit = np.arange(1, len(self.oos_pos)+1)

        for t in T_profit:

            # Produce a trading signal and calculate daily profit.
            daily_profit.append(trading_strategy(model_forecasts=self.avg_pred,
                                                 stock_price=close_prices,
                                                 k=1,
                                                 t=t))

        daily_profit = pd.Series(daily_profit)
        cumulative_profit = daily_profit.cumsum()
        annualised_return = get_annualised_return(
            cumulative_profit, T_profit)

        sharpe_ratio = get_sharpe_ratio(daily_profit)
        max_drawdown = get_max_dd_and_date(
            cumulative_profit)

        # Return the output.
        return daily_profit, cumulative_profit, annualised_return, sharpe_ratio, max_drawdown

    # def statistical_evaluation(self):


def DV_fun(sse, DV_pref=0.25):
    '''
    implement a middle of the range preference for middle of the range splits.
    '''

    seq = np.arange(1, len(sse)+1)
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
    sd_y = Y.std(axis=0, ddof=1)
    Y0 = (Y - np.repeat(mean_y,
                        repeats=size[0], axis=0)) / np.repeat(sd_y, repeats=size[0], axis=0)

    return {"Y": Y0, "mean": mean_y, "std": sd_y}

def get_sharpe_ratio(daily_profit):
    mean = daily_profit.mean()
    std_dev = daily_profit.std()
    return 252**(0.5)*mean/std_dev

def get_max_dd_and_date(cumulative_profit):
    rolling_max = (cumulative_profit+1).cummax()
    period_drawdown = (
        ((1+cumulative_profit)/rolling_max) - 1).astype(float)
    drawdown = round(period_drawdown.min(), 3)
    return drawdown

def get_annualised_return(cumulative_profit, T_profit):
    return cumulative_profit.iloc[-1]*(252/len(T_profit))


def trading_strategy(model_forecasts, stock_price, t, k=1):
    PL_t = 0
    signal_t_minus_1 = 0

    # Long if return prediction > 0 ; otherwise short.
    for i in range(1, k+1):
        if model_forecasts.iloc[t-i] > 0:
            signal_t_minus_1 += 1
        elif model_forecasts.iloc[t-i] < 0:
            signal_t_minus_1 -= 1
    PL_t += (1/k)*signal_t_minus_1 * \
        ((stock_price[t] - stock_price[t-1])/stock_price[t-1])

    return PL_t


def get_MAE(error_dict, model, T_model):
    abs_errors = map(abs, list(error_dict[model].values()))
    sum_abs_errors = sum(abs_errors)
    return (1/len(T_model))*sum_abs_errors


def get_MSE(error_dict, model, T_model):
    errors_as_array = np.array(list(error_dict[model].values()))
    sum_squared_errors = sum(np.power(errors_as_array, 2))
    return (1/len(T_model))*sum_squared_errors


def get_perc_correct(direction_correct_dict, model, T_model):
    amount_correct = direction_correct_dict[model]
    return 100*(1/len(T_model))*amount_correct

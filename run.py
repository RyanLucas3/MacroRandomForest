from line_profiler import LineProfiler
from MRF import *
data_in = pd.read_csv("/Users/ryanlucas/Desktop/MRF/MRF_data.csv")
MRF = MacroRandomForest(data=data_in, y_pos=0, x_pos=np.arange(1, 4), oos_pos=np.arange(
    150, 200), trend_push=4, quantile_rate=0.3, print_b=False, B=2)
mrf_output = MRF._ensemble_loop()

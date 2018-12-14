class GPLRV:
    def __init__(self):
        self.classes = 14
        self.class_names = [ 'class_6',
                             'class_15',
                             'class_16',
                             'class_42',
                             'class_52',
                             'class_53',
                             'class_62',
                             'class_64',
                             'class_65',
                             'class_67',
                             'class_88',
                             'class_90',
                             'class_92',
                             'class_95']


    def GrabPredictions(self, data):
        oof_preds = np.zeros((len(data), len(self.class_names)))
        oof_preds[:,0] = self.GP_class_6(data)
        oof_preds[:,1] = self.GP_class_15(data)
        oof_preds[:,2] = self.GP_class_16(data)
        oof_preds[:,3] = self.GP_class_42(data)
        oof_preds[:,4] = self.GP_class_52(data)
        oof_preds[:,5] = self.GP_class_53(data)
        oof_preds[:,6] = self.GP_class_62(data)
        oof_preds[:,7] = self.GP_class_64(data)
        oof_preds[:,8] = self.GP_class_65(data)
        oof_preds[:,9] = self.GP_class_67(data)
        oof_preds[:,10] = self.GP_class_88(data)
        oof_preds[:,11] = self.GP_class_90(data)
        oof_preds[:,12] = self.GP_class_92(data)
        oof_preds[:,13] = self.GP_class_95(data)
        oof_df = pd.DataFrame(np.exp(oof_preds), columns=self.class_names)
        oof_df =oof_df.div(oof_df.sum(axis=1), axis=0)
        return oof_df


    def GP_class_6(self,data):
        return (-1.965653 +
                0.100000*np.tanh(((((((((((data["flux_err_min"]) + ((((((data["0__kurtosis_y"]) < (data["flux_err_min"]))*1.)) * 2.0)))) / 2.0)) * 2.0)) * 2.0)) + (data["flux_err_min"]))) +
                0.100000*np.tanh(((((data["flux_diff"]) + (data["flux_err_min"]))) + (((data["flux_err_min"]) + (((((data["flux_err_min"]) * 2.0)) * 2.0)))))) +
                0.100000*np.tanh(((data["flux_diff"]) + (data["flux_err_min"]))) +
                0.100000*np.tanh(((((((data["detected_flux_err_mean"]) * 2.0)) + (data["flux_err_min"]))) + (data["flux_err_median"]))) +
                0.100000*np.tanh(((data["flux_std"]) + (((data["4__skewness_x"]) + ((((((data["4__skewness_x"]) + (data["detected_flux_err_mean"]))) + (data["detected_flux_err_median"]))/2.0)))))) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["flux_mean"])), ((((((data["flux_err_median"]) * 2.0)) * 2.0)))))), ((data["flux_err_mean"])))) + (data["flux_err_mean"]))) +
                0.100000*np.tanh(((np.minimum(((data["5__kurtosis_x"])), ((((data["flux_err_median"]) * 2.0))))) + (((data["flux_err_min"]) / 2.0)))) +
                0.100000*np.tanh(((data["flux_err_min"]) + (((data["3__fft_coefficient__coeff_0__attr__abs__x"]) + (((((np.minimum(((data["flux_err_min"])), ((data["detected_flux_min"])))) * 2.0)) * 2.0)))))) +
                0.100000*np.tanh(np.minimum(((np.where(data["4__kurtosis_x"] > -1, data["detected_flux_skew"], ((np.minimum(((data["detected_flux_skew"])), ((data["detected_flux_err_min"])))) * 2.0) ))), ((((data["detected_flux_err_min"]) + (data["detected_flux_skew"])))))) +
                0.100000*np.tanh(((np.minimum(((data["5__kurtosis_x"])), ((((((data["flux_err_min"]) * 2.0)) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]<0, data["5__kurtosis_x"], ((data["5__kurtosis_x"]) * 2.0) ))), ((data["5__fft_coefficient__coeff_0__attr__abs__y"])))) * 2.0)) +
                0.100000*np.tanh((((data["detected_flux_err_min"]) + (((np.where((((5.20438098907470703)) + (data["detected_flux_err_median"]))>0, data["detected_flux_err_min"], data["detected_flux_err_min"] )) * 2.0)))/2.0)) +
                0.100000*np.tanh(((data["detected_flux_err_min"]) + (((np.minimum(((((((-1.0) - (data["detected_flux_min"]))) - (data["distmod"])))), ((data["detected_flux_min"])))) - (data["distmod"]))))) +
                0.100000*np.tanh(np.where(data["flux_err_min"]>0, data["5__kurtosis_x"], np.where(data["mwebv"] > -1, np.minimum(((((data["detected_flux_min"]) * 2.0))), ((((data["flux_err_min"]) * 2.0)))), data["flux_err_min"] ) )) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["detected_flux_min"])), ((data["flux_err_min"]))))), ((data["flux_err_min"])))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["flux_skew"])), ((((data["flux_err_min"]) * (((data["detected_flux_min"]) * (data["flux_err_min"])))))))) + (((data["flux_err_min"]) + (data["flux_err_min"]))))) +
                0.100000*np.tanh(((((data["4__skewness_x"]) + (np.minimum(((data["flux_err_min"])), ((np.minimum(((data["flux_err_min"])), ((data["4__skewness_x"]))))))))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((data["4__skewness_x"])), ((((((((data["flux_err_min"]) * 2.0)) * 2.0)) * 2.0))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["5__kurtosis_x"])), ((np.minimum(((data["flux_err_min"])), ((data["detected_flux_err_std"]))))))) +
                0.100000*np.tanh(((np.minimum(((data["5__kurtosis_x"])), ((data["flux_err_min"])))) * 2.0)) +
                0.100000*np.tanh(((((np.where(data["flux_err_min"]<0, data["flux_err_min"], ((data["flux_err_min"]) * 2.0) )) * 2.0)) + (np.minimum(((data["flux_err_min"])), ((data["detected_flux_min"])))))) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (data["flux_err_min"]))) * 2.0)) +
                0.100000*np.tanh(((data["flux_d1_pb5"]) * (data["detected_flux_max"]))) +
                0.100000*np.tanh(np.where(np.where(data["hostgal_photoz"] > -1, data["detected_flux_err_max"], data["5__kurtosis_x"] ) > -1, np.where(data["distmod"] > -1, data["distmod"], data["5__skewness_x"] ), data["5__kurtosis_x"] )) +
                0.100000*np.tanh(((data["flux_max"]) * (np.minimum(((np.minimum(((((data["detected_flux_min"]) + (data["detected_flux_skew"])))), ((data["flux_max"]))))), ((data["detected_flux_min"])))))) +
                0.100000*np.tanh(((data["detected_flux_min"]) - (data["distmod"]))) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, -3.0, np.where(data["hostgal_photoz"] > -1, data["distmod"], (((-1.0*((-3.0)))) + (data["detected_flux_min"])) ) )) +
                0.100000*np.tanh(np.minimum(((data["flux_err_min"])), ((data["flux_err_min"])))) +
                0.100000*np.tanh(((data["flux_err_min"]) + (data["flux_err_min"]))) +
                0.100000*np.tanh(((np.minimum(((data["detected_flux_skew"])), ((data["5__skewness_x"])))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((((data["flux_err_min"]) * 2.0))), ((((((((np.minimum(((data["flux_err_min"])), ((data["detected_flux_min"])))) * 2.0)) * 2.0)) * 2.0))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(((data["distmod"]) / 2.0) > -1, np.where(np.where(data["distmod"] > -1, data["distmod"], 3.141593 ) > -1, -3.0, data["distmod"] ), data["5__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(((np.where(-2.0 > -1, ((-2.0) - (data["distmod"])), ((-2.0) - (data["distmod"])) )) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_min"]) + (((((np.minimum(((data["flux_d0_pb4"])), ((data["detected_flux_max"])))) + (np.minimum(((data["flux_max"])), ((data["flux_err_min"])))))) * (data["flux_max"]))))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) * (((np.minimum(((((data["flux_d0_pb4"]) * (data["detected_flux_max"])))), ((data["detected_flux_skew"])))) * (data["detected_flux_max"]))))) - (data["detected_flux_max"]))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (np.minimum(((data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((data["detected_flux_skew"])))))) * (((data["detected_flux_min"]) + (data["detected_flux_skew"]))))) +
                0.100000*np.tanh(np.where((((data["distmod"]) + (data["distmod"]))/2.0) > -1, -1.0, np.where(((data["distmod"]) / 2.0) > -1, -1.0, (-1.0*((data["distmod"]))) ) )) +
                0.100000*np.tanh(((((np.minimum(((((((-2.0) * 2.0)) - (data["distmod"])))), ((-2.0)))) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["flux_err_min"])), ((np.minimum(((data["5__skewness_x"])), ((((data["flux_err_min"]) + ((((((-1.0) * 2.0)) + (data["detected_flux_skew"]))/2.0)))))))))) +
                0.100000*np.tanh(np.where(data["flux_d0_pb4"]<0, (((data["flux_by_flux_ratio_sq_skew"]) + (((data["detected_flux_max"]) * (data["flux_d0_pb4"]))))/2.0), data["detected_flux_max"] )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, np.where(-3.0 > -1, -3.0, -3.0 ), ((data["4__skewness_x"]) - (data["hostgal_photoz"])) )) +
                0.100000*np.tanh(((((((((-3.0) - (data["distmod"]))) * 2.0)) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh((-1.0*((np.where(((data["distmod"]) - (-2.0))<0, -2.0, ((((data["distmod"]) - (-2.0))) - (data["distmod"])) ))))) +
                0.100000*np.tanh((-1.0*((((data["distmod"]) + (((((((data["detected_mjd_diff"]) + (data["distmod"]))) + (3.141593))) + (data["distmod"])))))))) +
                0.100000*np.tanh(((((((-1.0) - (data["distmod"]))) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, ((np.where(data["hostgal_photoz"] > -1, data["hostgal_photoz"], data["5__fft_coefficient__coeff_0__attr__abs__y"] )) - (3.141593)), ((data["detected_flux_min"]) * (data["flux_max"])) )) +
                0.100000*np.tanh(np.where(3.141593 > -1, (-1.0*((np.where(((data["distmod"]) / 2.0) > -1, 2.0, data["distmod"] )))), ((-1.0) - (data["distmod"])) )) +
                0.100000*np.tanh(((((((data["flux_d0_pb4"]) * (data["flux_max"]))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((np.minimum(((((-2.0) - (data["distmod"])))), ((-1.0)))) - (data["distmod"]))) +
                0.100000*np.tanh(((data["detected_flux_max"]) * (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) * (((data["flux_max"]) * (((data["detected_flux_min"]) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))))))) +
                0.100000*np.tanh(((np.minimum((((((-1.0*(((((2.0) + (data["distmod"]))/2.0))))) * 2.0))), ((((data["detected_flux_err_std"]) * 2.0))))) * 2.0)) +
                0.100000*np.tanh((((((((data["4__skewness_y"]) * (((((data["0__skewness_x"]) * 2.0)) * 2.0)))) + (((data["0__skewness_x"]) - (data["detected_mjd_diff"]))))/2.0)) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.where(((data["mwebv"]) - (data["flux_err_min"])) > -1, ((-3.0) * (((data["mwebv"]) - (data["flux_err_min"])))), data["flux_skew"] )) +
                0.100000*np.tanh(np.where((((data["detected_flux_err_std"]) + (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (data["5__fft_coefficient__coeff_0__attr__abs__y"]))))/2.0)>0, data["5__fft_coefficient__coeff_0__attr__abs__y"], np.tanh((data["5__fft_coefficient__coeff_0__attr__abs__y"])) )) +
                0.100000*np.tanh(((((((((np.minimum(((-2.0)), ((-2.0)))) - (data["distmod"]))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_min"]) * (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (data["4__skewness_y"]))) + ((((3.0)) * (((data["4__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) * (data["detected_flux_min"]))))))))) +
                0.100000*np.tanh(((((((data["detected_flux_median"]) * (data["detected_flux_max"]))) + (data["4__skewness_y"]))) + (((((data["flux_min"]) * (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))))) +
                0.100000*np.tanh(np.where(data["detected_flux_max"] > -1, ((data["5__skewness_y"]) - (data["detected_mjd_diff"])), ((data["detected_mjd_diff"]) - (data["detected_flux_max"])) )) +
                0.100000*np.tanh(((data["detected_flux_ratio_sq_skew"]) + (np.minimum(((data["5__kurtosis_x"])), (((((data["4__skewness_x"]) + (data["detected_flux_by_flux_ratio_sq_skew"]))/2.0))))))) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_err_skew"])), ((((((((data["detected_flux_err_skew"]) + (data["3__skewness_y"]))) + (data["detected_flux_err_skew"]))) + (((data["3__skewness_y"]) + (data["detected_flux_err_skew"])))))))) +
                0.100000*np.tanh((((((-1.0*((np.where(np.where(data["detected_flux_by_flux_ratio_sq_skew"]<0, data["flux_err_min"], data["3__fft_coefficient__coeff_1__attr__abs__y"] ) > -1, data["distmod"], data["detected_flux_err_std"] ))))) - (data["detected_flux_err_std"]))) * 2.0)) +
                0.100000*np.tanh(((((((((np.where(data["detected_mjd_diff"] > -1, data["3__skewness_y"], data["detected_mjd_diff"] )) - (data["detected_mjd_diff"]))) * 2.0)) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((np.minimum(((data["flux_err_min"])), ((data["detected_flux_err_std"]))))), ((data["flux_err_min"])))) + (data["flux_err_min"]))) * 2.0)) +
                0.100000*np.tanh((((data["flux_d0_pb4"]) + (data["flux_d0_pb4"]))/2.0)) +
                0.100000*np.tanh(((3.0) - (np.where(((data["distmod"]) / 2.0) > -1, (14.57934379577636719), data["hostgal_photoz"] )))) +
                0.100000*np.tanh(((data["flux_min"]) + (data["flux_err_min"]))) +
                0.100000*np.tanh(((np.where(data["mwebv"] > -1, ((data["flux_err_min"]) - (data["mwebv"])), data["2__kurtosis_x"] )) * 2.0)) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__y"]) - (np.where((6.0) > -1, np.where(data["hostgal_photoz"] > -1, (6.0), data["hostgal_photoz"] ), data["1__fft_coefficient__coeff_0__attr__abs__y"] )))) +
                0.100000*np.tanh(((np.minimum(((data["1__skewness_x"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.where(((data["distmod"]) / 2.0) > -1, data["distmod"], np.maximum(((((data["detected_flux_err_std"]) / 2.0))), ((np.maximum(((data["detected_flux_err_std"])), ((data["distmod"])))))) )) +
                0.100000*np.tanh(((np.where(data["4__fft_coefficient__coeff_0__attr__abs__y"] > -1, data["5__skewness_y"], data["detected_mjd_diff"] )) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((np.where(data["4__fft_coefficient__coeff_0__attr__abs__y"] > -1, ((np.where(data["hostgal_photoz"] > -1, data["hostgal_photoz"], data["4__fft_coefficient__coeff_0__attr__abs__y"] )) * 2.0), data["4__fft_coefficient__coeff_0__attr__abs__y"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_err_max"]>0, data["detected_flux_err_skew"], np.where(data["flux_max"]>0, np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"]>0, data["detected_flux_err_skew"], ((data["detected_flux_err_skew"]) * 2.0) ), data["detected_flux_err_max"] ) )) +
                0.100000*np.tanh((-1.0*((((2.718282) + (((data["distmod"]) + (data["distmod"])))))))) +
                0.100000*np.tanh(((((data["1__kurtosis_x"]) - (data["detected_flux_max"]))) + (((data["flux_min"]) + (np.minimum(((data["1__skewness_x"])), ((data["1__skewness_x"])))))))) +
                0.100000*np.tanh((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"]>0, data["5__fft_coefficient__coeff_1__attr__abs__x"], data["5__fft_coefficient__coeff_0__attr__abs__y"] )))/2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_max"] > -1, ((np.where(data["detected_flux_max"] > -1, data["detected_flux_min"], ((data["detected_flux_err_median"]) + (0.367879)) )) + (data["2__skewness_x"])), data["detected_flux_err_max"] )) +
                0.100000*np.tanh(((np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"]<0, data["4__fft_coefficient__coeff_0__attr__abs__y"], data["detected_flux_err_max"] )) + (((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) / 2.0)) + (data["2__kurtosis_x"]))))) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) * (((np.where(data["2__fft_coefficient__coeff_0__attr__abs__y"] > -1, data["flux_d1_pb5"], np.where(data["2__skewness_y"]<0, data["flux_d0_pb0"], data["flux_std"] ) )) - (data["flux_d0_pb0"]))))) +
                0.100000*np.tanh(((((data["detected_flux_err_mean"]) * 2.0)) + (data["detected_flux_err_max"]))) +
                0.100000*np.tanh(((((((data["flux_err_min"]) - (np.where(np.tanh((data["1__fft_coefficient__coeff_1__attr__abs__y"])) > -1, data["mwebv"], data["mwebv"] )))) * 2.0)) * 2.0)) +
                0.100000*np.tanh((((((data["4__skewness_y"]) + (np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["1__kurtosis_x"], data["flux_min"] )))/2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["1__kurtosis_x"]>0, (((data["3__skewness_y"]) < (data["5__skewness_y"]))*1.), (((data["flux_median"]) + (((data["detected_flux_w_mean"]) + (((data["1__kurtosis_x"]) * 2.0)))))/2.0) )) +
                0.100000*np.tanh(np.maximum(((np.where(data["hostgal_photoz"] > -1, -2.0, np.maximum(((((data["flux_min"]) + (data["flux_err_min"])))), ((-2.0))) ))), ((-2.0)))) +
                0.100000*np.tanh(((((data["5__skewness_y"]) + (data["flux_err_min"]))) - (np.where(((data["detected_mjd_diff"]) + (((data["5__skewness_y"]) / 2.0))) > -1, data["detected_mjd_diff"], data["detected_mjd_diff"] )))) +
                0.100000*np.tanh(((data["flux_err_min"]) + (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.minimum(((data["flux_d1_pb0"])), ((((np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]<0, data["1__kurtosis_x"], np.minimum(((data["detected_flux_err_skew"])), ((data["1__kurtosis_x"]))) )) * 2.0))))) +
                0.100000*np.tanh(((np.maximum(((((np.tanh((data["flux_d1_pb0"]))) * (data["mjd_diff"])))), ((data["detected_flux_err_skew"])))) * (data["2__skewness_x"]))) +
                0.100000*np.tanh(np.where(data["0__skewness_x"]<0, data["flux_d1_pb1"], np.tanh((((np.maximum(((data["detected_flux_err_max"])), ((((data["4__skewness_x"]) - (data["flux_err_skew"])))))) + (data["4__skewness_x"])))) )) +
                0.100000*np.tanh(((1.0) * (data["detected_flux_err_median"]))) +
                0.100000*np.tanh(((np.minimum(((((np.tanh((data["5__skewness_y"]))) * (data["2__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["flux_by_flux_ratio_sq_sum"])))) * (np.where(data["1__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_ratio_sq_skew"], data["2__skewness_y"] )))) +
                0.100000*np.tanh(((((((-2.0) - (data["distmod"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh((((((-1.0*((((data["distmod"]) + (np.maximum(((2.0)), ((((data["distmod"]) + (data["distmod"]))))))))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((((np.where(data["4__fft_coefficient__coeff_0__attr__abs__y"] > -1, ((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0), data["4__fft_coefficient__coeff_0__attr__abs__y"] )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_d1_pb2"]>0, ((data["1__skewness_x"]) * 2.0), np.where(((data["flux_dif2"]) * 2.0)<0, ((data["detected_flux_max"]) * (data["detected_flux_dif3"])), data["flux_err_skew"] ) )) +
                0.100000*np.tanh((((data["detected_flux_by_flux_ratio_sq_skew"]) + (data["1__kurtosis_x"]))/2.0)) +
                0.100000*np.tanh(((((data["flux_d1_pb0"]) * (data["4__fft_coefficient__coeff_0__attr__abs__y"]))) * (data["3__skewness_y"]))) +
                0.100000*np.tanh(np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__y"], np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"] > -1, data["5__skewness_y"], ((data["4__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0) ) )) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, (((((data["3__kurtosis_y"]) + (data["flux_err_min"]))/2.0)) - (data["detected_mjd_diff"])), data["detected_mjd_diff"] )) +
                0.100000*np.tanh(((data["flux_d0_pb4"]) * (data["3__skewness_x"]))) +
                0.100000*np.tanh(np.where(((-2.0) - (data["distmod"]))<0, np.where(-2.0<0, ((-2.0) - (data["distmod"])), data["detected_flux_median"] ), 2.718282 )) +
                0.100000*np.tanh(((data["flux_median"]) + (((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_min"]))) + (data["flux_min"]))))) +
                0.100000*np.tanh(((((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) > (data["0__kurtosis_x"]))*1.)) + (np.minimum(((data["flux_err_min"])), ((data["flux_err_min"])))))/2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_err_skew"]<0, data["detected_flux_skew"], np.minimum(((data["flux_err_min"])), ((data["flux_err_min"]))) )) +
                0.100000*np.tanh(((data["0__kurtosis_x"]) + ((((((data["0__kurtosis_x"]) + (((data["3__skewness_y"]) + (data["3__fft_coefficient__coeff_1__attr__abs__x"]))))/2.0)) + (data["detected_flux_min"]))))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_0__attr__abs__x"]) - (((data["detected_mjd_diff"]) / 2.0)))) +
                0.100000*np.tanh(((((data["detected_flux_by_flux_ratio_sq_sum"]) * (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) * (data["detected_flux_by_flux_ratio_sq_sum"]))) +
                0.100000*np.tanh(((((((data["detected_mean"]) - (data["mwebv"]))) - (data["mwebv"]))) - (data["mwebv"]))) +
                0.100000*np.tanh(np.where(data["5__skewness_x"]<0, data["flux_err_skew"], ((data["1__kurtosis_x"]) * ((((data["detected_flux_ratio_sq_sum"]) > (np.maximum(((data["flux_mean"])), ((data["4__fft_coefficient__coeff_0__attr__abs__y"])))))*1.))) )) +
                0.100000*np.tanh(((np.maximum(((data["flux_median"])), ((data["flux_median"])))) * (((data["5__skewness_x"]) / 2.0)))) +
                0.100000*np.tanh(np.where(np.tanh((2.718282)) > -1, np.where(data["detected_flux_by_flux_ratio_sq_sum"]<0, np.where(data["flux_err_skew"] > -1, data["detected_flux_err_skew"], data["detected_flux_skew"] ), data["5__fft_coefficient__coeff_0__attr__abs__x"] ), data["mjd_diff"] )) +
                0.100000*np.tanh(((((data["detected_flux_err_skew"]) * (np.tanh((data["detected_flux_err_skew"]))))) * (((data["4__skewness_x"]) * (data["detected_flux_err_skew"]))))) +
                0.100000*np.tanh(((((np.where(data["1__fft_coefficient__coeff_1__attr__abs__y"]<0, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["5__fft_coefficient__coeff_1__attr__abs__x"] )) * 2.0)) * (((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["1__fft_coefficient__coeff_1__attr__abs__y"]))))) +
                0.100000*np.tanh(np.where(data["mwebv"]<0, data["flux_d1_pb2"], ((data["flux_err_min"]) - (data["mwebv"])) )) +
                0.100000*np.tanh(np.where(data["flux_err_skew"]<0, ((((data["flux_err_skew"]) + (data["flux_err_skew"]))) + (data["detected_flux_median"])), ((data["detected_flux_median"]) + ((-1.0*((data["detected_flux_median"]))))) )) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"] > -1, np.where(data["flux_d1_pb0"]>0, data["detected_flux_err_median"], np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]>0, data["flux_err_min"], data["flux_err_min"] ) ), data["flux_skew"] )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb0"] > -1, ((data["4__fft_coefficient__coeff_1__attr__abs__y"]) / 2.0), np.tanh((data["detected_flux_ratio_sq_sum"])) )) +
                0.100000*np.tanh(((np.where(data["detected_mjd_diff"] > -1, data["flux_err_min"], data["flux_err_min"] )) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.where(data["0__skewness_x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__y"], data["4__fft_coefficient__coeff_1__attr__abs__y"] )))

    def GP_class_15(self,data):
        return (-1.349153 +
                0.100000*np.tanh(((data["flux_d1_pb0"]) + (np.minimum(((data["0__skewness_x"])), ((((((data["0__skewness_x"]) - (data["mjd_size"]))) + (data["distmod"])))))))) +
                0.100000*np.tanh(((((data["flux_d1_pb0"]) + (np.minimum(((((data["flux_d0_pb1"]) + (((data["flux_d1_pb0"]) + (data["flux_d1_pb0"])))))), ((data["5__kurtosis_y"])))))) * 2.0)) +
                0.100000*np.tanh(((((data["flux_d0_pb0"]) + (data["distmod"]))) + (((((np.minimum(((data["0__skewness_x"])), ((data["flux_d0_pb0"])))) - (data["detected_mjd_size"]))) * 2.0)))) +
                0.100000*np.tanh(np.minimum(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (((data["distmod"]) + (((data["detected_flux_min"]) * 2.0))))))), ((data["flux_d0_pb0"])))) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["flux_d1_pb0"])), ((data["flux_d0_pb0"]))))), ((((np.minimum(((data["0__skewness_x"])), ((data["detected_flux_min"])))) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((((np.minimum(((data["0__skewness_x"])), ((((data["flux_d1_pb0"]) * 2.0))))) * 2.0))), ((((data["flux_d1_pb0"]) * 2.0))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["flux_d0_pb1"])), ((np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__x"])), (((-1.0*((data["5__fft_coefficient__coeff_0__attr__abs__y"])))))))))) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((((data["distmod"]) + (((data["0__skewness_x"]) + (data["detected_flux_min"]))))) + (data["flux_err_std"]))) + (data["detected_flux_min"]))) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (((np.minimum(((data["0__skewness_x"])), ((data["flux_d0_pb0"])))) + (((data["flux_d0_pb0"]) + (data["distmod"]))))))) +
                0.100000*np.tanh(np.minimum(((((data["flux_d0_pb0"]) + (data["5__kurtosis_y"])))), ((data["flux_d1_pb0"])))) +
                0.100000*np.tanh(((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__y"])), ((((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["1__skewness_x"]) + (data["5__fft_coefficient__coeff_1__attr__abs__y"]))))) + (data["1__skewness_x"])))))) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((data["flux_d1_pb1"]) - (1.0))) - (((data["5__fft_coefficient__coeff_1__attr__abs__y"]) - (((((data["flux_d1_pb0"]) - (data["ddf"]))) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))))))) +
                0.100000*np.tanh(((((((data["2__kurtosis_x"]) + (np.tanh((data["0__skewness_x"]))))) + (data["detected_flux_min"]))) + (np.where(data["distmod"] > -1, data["ddf"], data["detected_flux_min"] )))) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) + (((((data["distmod"]) + (((data["0__skewness_x"]) + (data["detected_flux_min"]))))) + (data["distmod"]))))) +
                0.100000*np.tanh((((((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) + (data["flux_d0_pb1"]))/2.0)) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) + (np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__x"])), ((data["detected_flux_min"])))))) +
                0.100000*np.tanh(((((((data["flux_d0_pb0"]) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((data["1__skewness_x"]) + (((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["detected_flux_min"]) * 2.0)))) + (np.tanh((((data["distmod"]) + (data["flux_ratio_sq_skew"]))))))))) +
                0.100000*np.tanh(((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((data["hostgal_photoz_err"])))) + (((((data["detected_flux_min"]) + (((data["distmod"]) + (data["mjd_diff"]))))) + (data["detected_flux_min"]))))) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(np.minimum(((((data["flux_ratio_sq_skew"]) + (np.minimum(((data["detected_flux_by_flux_ratio_sq_skew"])), ((data["flux_ratio_sq_skew"]))))))), ((((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (data["distmod"]))) + (data["flux_ratio_sq_skew"])))))) +
                0.100000*np.tanh(((((((np.minimum(((data["flux_ratio_sq_skew"])), ((data["distmod"])))) + (data["flux_ratio_sq_skew"]))) + (data["1__skewness_x"]))) + (np.minimum(((data["flux_err_median"])), ((data["flux_ratio_sq_skew"])))))) +
                0.100000*np.tanh(((((data["flux_d0_pb0"]) - (data["4__kurtosis_x"]))) + (np.minimum(((data["flux_d1_pb1"])), ((np.minimum(((data["flux_d0_pb0"])), ((np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__x"])), ((data["0__fft_coefficient__coeff_1__attr__abs__x"])))))))))))) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) - (np.where(data["3__fft_coefficient__coeff_1__attr__abs__x"]<0, np.where(data["detected_mjd_size"] > -1, data["1__fft_coefficient__coeff_0__attr__abs__y"], data["flux_d0_pb0"] ), data["detected_mjd_size"] )))) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) + (data["distmod"]))) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((((((data["flux_d0_pb0"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) + ((((((data["flux_ratio_sq_skew"]) + (np.minimum(((data["flux_err_std"])), ((data["distmod"])))))/2.0)) + (data["flux_err_median"]))))) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) + (np.minimum(((np.minimum(((((data["distmod"]) - (0.0)))), ((data["flux_d1_pb1"]))))), ((data["distmod"])))))) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), ((data["flux_ratio_sq_skew"]))))), ((data["1__skewness_x"]))))), ((data["mjd_diff"]))))), ((data["flux_ratio_sq_skew"])))) +
                0.100000*np.tanh(((data["distmod"]) + (((((data["mjd_diff"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) + (data["distmod"]))))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) + (((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(((data["flux_d1_pb1"]) + (((((data["distmod"]) + (data["detected_flux_min"]))) - (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (np.minimum(((data["detected_flux_min"])), ((data["distmod"])))))))))) +
                0.100000*np.tanh(((((((data["flux_ratio_sq_skew"]) + (data["distmod"]))) + (data["1__skewness_x"]))) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((data["detected_flux_by_flux_ratio_sq_skew"]) + (np.minimum(((((data["detected_flux_err_min"]) + (data["1__fft_coefficient__coeff_0__attr__abs__x"])))), ((data["detected_flux_ratio_sq_skew"])))))) +
                0.100000*np.tanh(((((((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["flux_d0_pb1"]) + (((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__y"]))))) + (((data["distmod"]) + (data["distmod"]))))) +
                0.100000*np.tanh(((((((data["4__skewness_x"]) - (data["ddf"]))) - (((data["4__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_err_mean"]))))) - (data["ddf"]))) +
                0.100000*np.tanh(((data["flux_d0_pb1"]) - (np.where(((data["5__fft_coefficient__coeff_0__attr__abs__x"]) * (data["flux_d0_pb1"]))>0, data["5__fft_coefficient__coeff_1__attr__abs__y"], data["1__skewness_x"] )))) +
                0.100000*np.tanh(((((((data["detected_flux_ratio_sq_skew"]) - (data["ddf"]))) + (data["detected_flux_by_flux_ratio_sq_skew"]))) + (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(((((((((data["flux_d1_pb1"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) + (((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__y"]))))) - (data["ddf"]))) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((data["4__kurtosis_x"]) + (data["detected_flux_ratio_sq_skew"]))) + (((data["distmod"]) + (data["4__kurtosis_x"]))))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["flux_d0_pb1"]) - (data["3__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(((data["flux_d0_pb1"]) + (np.minimum(((data["flux_err_mean"])), ((data["distmod"])))))) +
                0.100000*np.tanh(((((((data["1__skewness_x"]) - (((data["0__fft_coefficient__coeff_0__attr__abs__x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))))) + (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))))) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_by_flux_ratio_sq_skew"])), ((((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), ((data["detected_flux_by_flux_ratio_sq_skew"])))) + (data["4__skewness_x"])))))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_ratio_sq_skew"]) - (data["flux_d0_pb0"]))) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) + (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((np.where(data["4__kurtosis_x"]>0, np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]>0, ((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"])), data["3__kurtosis_x"] ), data["3__skewness_x"] )) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["flux_d0_pb1"]) + (((data["0__fft_coefficient__coeff_1__attr__abs__y"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))))) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((data["ddf"]) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((data["detected_flux_ratio_sq_skew"]) + (((data["distmod"]) + (((data["mjd_diff"]) + (((data["flux_d0_pb0"]) + (data["mjd_diff"]))))))))) * 2.0)) +
                0.100000*np.tanh(((((np.tanh((data["4__kurtosis_x"]))) * (((((data["detected_flux_min"]) * (data["4__kurtosis_x"]))) + (((data["detected_flux_min"]) * (data["4__kurtosis_x"]))))))) * 2.0)) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) - (data["ddf"]))) + (((((data["flux_ratio_sq_skew"]) - (((data["ddf"]) - (data["detected_flux_min"]))))) - (data["2__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(((data["2__skewness_x"]) + (((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (((np.minimum(((data["hostgal_photoz_err"])), ((data["distmod"])))) + (((data["hostgal_photoz_err"]) + (data["detected_flux_min"]))))))))) +
                0.100000*np.tanh(np.where(np.where(data["detected_flux_err_min"]<0, data["detected_flux_ratio_sq_skew"], data["5__kurtosis_x"] )<0, ((data["detected_flux_err_min"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])), data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((data["detected_flux_ratio_sq_sum"])), ((data["3__kurtosis_x"]))))), ((((data["distmod"]) + (data["flux_ratio_sq_skew"]))))))), ((data["detected_flux_min"])))) +
                0.100000*np.tanh(np.where((((data["flux_d0_pb0"]) < (data["flux_ratio_sq_sum"]))*1.)>0, data["detected_flux_err_min"], 0.367879 )) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) * 2.0)) + (((((data["detected_mjd_diff"]) * 2.0)) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (data["distmod"]))) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh((((data["detected_mean"]) < (((data["flux_d1_pb0"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))))*1.)) +
                0.100000*np.tanh(((((data["flux_d0_pb0"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["detected_mean"]>0, np.where(data["hostgal_photoz_err"]>0, data["distmod"], np.where(data["detected_mean"]>0, data["distmod"], data["distmod"] ) ), data["flux_ratio_sq_sum"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_d0_pb1"], (((-1.0*((data["4__fft_coefficient__coeff_1__attr__abs__y"])))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"])) )) +
                0.100000*np.tanh(((((data["2__fft_coefficient__coeff_0__attr__abs__x"]) + (((((data["detected_flux_by_flux_ratio_sq_skew"]) + (data["flux_ratio_sq_sum"]))) + (data["flux_d1_pb5"]))))) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((-1.0*((data["3__fft_coefficient__coeff_0__attr__abs__y"])))) > (data["detected_flux_err_min"]))*1.)) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((data["flux_d1_pb3"]) * (np.where(np.minimum(((data["ddf"])), ((data["5__skewness_x"]))) > -1, data["2__kurtosis_x"], np.where(data["5__skewness_x"] > -1, data["4__kurtosis_x"], data["5__skewness_x"] ) )))) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]>0, ((((data["flux_d0_pb0"]) - (data["detected_flux_std"]))) * 2.0), ((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["flux_ratio_sq_sum"]) * 2.0))) )) +
                0.100000*np.tanh(((((((((data["flux_d0_pb0"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_diff"]>0, np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["4__kurtosis_x"], np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["4__kurtosis_x"] ) ), data["1__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(((np.where(data["detected_flux_std"]>0, data["detected_mjd_diff"], ((data["2__kurtosis_x"]) + (data["1__fft_coefficient__coeff_1__attr__abs__y"])) )) + (np.where(data["2__kurtosis_x"]>0, data["2__kurtosis_x"], data["flux_d0_pb5"] )))) +
                0.100000*np.tanh(((np.where(np.minimum(((data["detected_flux_err_min"])), ((data["detected_flux_ratio_sq_sum"])))>0, data["4__kurtosis_x"], data["distmod"] )) * (data["detected_flux_std"]))) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]<0, data["mjd_diff"], np.maximum(((data["flux_dif2"])), ((((data["2__kurtosis_x"]) - (np.minimum(((data["flux_dif2"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))))))) )) +
                0.100000*np.tanh((((data["distmod"]) + (np.where(data["detected_flux_err_min"]>0, data["5__skewness_x"], ((np.minimum(((data["5__skewness_x"])), ((data["3__kurtosis_x"])))) + (data["flux_d0_pb5"])) )))/2.0)) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (data["1__kurtosis_x"]))) +
                0.100000*np.tanh((((((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) < (data["flux_median"]))*1.)) + (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) + (data["flux_d1_pb2"]))) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_0__attr__abs__x"]<0, data["detected_flux_std"], ((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (data["detected_flux_std"])) )) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]<0, np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, data["2__fft_coefficient__coeff_1__attr__abs__x"], data["flux_ratio_sq_sum"] ), data["detected_mjd_diff"] )) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_err_min"]))) + (data["detected_flux_err_min"]))) +
                0.100000*np.tanh((((((data["hostgal_photoz_err"]) * 2.0)) + (np.where((((data["flux_ratio_sq_skew"]) + ((((data["hostgal_photoz_err"]) + (data["distmod"]))/2.0)))/2.0)>0, data["flux_ratio_sq_skew"], data["hostgal_photoz_err"] )))/2.0)) +
                0.100000*np.tanh(((np.where(np.where(data["4__kurtosis_x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__y"], data["4__fft_coefficient__coeff_1__attr__abs__y"] )<0, data["1__fft_coefficient__coeff_1__attr__abs__y"], data["detected_mjd_diff"] )) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_d0_pb1"]) + (((data["flux_ratio_sq_skew"]) - (data["mwebv"]))))) - (data["ddf"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((np.where((((data["4__kurtosis_x"]) + (data["3__kurtosis_x"]))/2.0)<0, ((data["4__kurtosis_x"]) - (data["detected_flux_skew"])), data["flux_d1_pb5"] )) * (data["detected_flux_skew"]))) +
                0.100000*np.tanh(((data["detected_flux_min"]) + (((data["detected_flux_min"]) + (((np.minimum(((data["distmod"])), ((data["distmod"])))) * 2.0)))))) +
                0.100000*np.tanh(((((((np.where((-1.0*((data["detected_mjd_diff"]))) > -1, data["detected_mjd_diff"], (-1.0*((data["detected_mjd_diff"]))) )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(data["detected_flux_err_min"]>0, data["flux_d1_pb5"], ((((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_flux_max"]))) * 2.0)) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(np.minimum((((((((data["flux_median"]) * 2.0)) + (data["flux_median"]))/2.0))), ((data["mjd_size"])))) +
                0.100000*np.tanh(np.where(data["2__skewness_x"]<0, np.maximum(((((((data["flux_d0_pb1"]) * 2.0)) + (np.maximum(((data["flux_d0_pb1"])), ((data["2__kurtosis_x"]))))))), ((data["detected_flux_ratio_sq_sum"]))), data["flux_d0_pb1"] )) +
                0.100000*np.tanh(((data["2__kurtosis_x"]) + (np.maximum(((((data["1__kurtosis_y"]) + (data["2__kurtosis_x"])))), ((((data["2__kurtosis_x"]) + (data["2__kurtosis_x"])))))))) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]<0, data["detected_flux_ratio_sq_sum"], ((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"])) )) +
                0.100000*np.tanh(((((((((data["detected_mjd_diff"]) * (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) + (data["detected_flux_err_min"]))) + ((((data["detected_flux_max"]) < (data["detected_flux_min"]))*1.)))) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((np.where(data["detected_flux_err_min"]<0, ((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"])), ((data["detected_flux_dif2"]) * 2.0) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (((((data["flux_err_mean"]) - (np.where(data["flux_d1_pb1"]>0, data["3__fft_coefficient__coeff_1__attr__abs__y"], data["4__kurtosis_y"] )))) + (data["5__skewness_x"]))))) +
                0.100000*np.tanh(((((((data["distmod"]) + (data["detected_flux_ratio_sq_sum"]))) + (data["detected_mjd_diff"]))) + (((data["2__fft_coefficient__coeff_0__attr__abs__x"]) + (np.minimum(((data["flux_ratio_sq_sum"])), ((data["3__kurtosis_x"])))))))) +
                0.100000*np.tanh(((np.where(data["flux_d1_pb5"]<0, np.where(data["1__kurtosis_y"]<0, data["1__kurtosis_y"], data["distmod"] ), data["1__fft_coefficient__coeff_1__attr__abs__y"] )) + (data["detected_flux_min"]))) +
                0.100000*np.tanh(((np.where(data["detected_flux_err_min"]>0, data["flux_ratio_sq_sum"], np.where(data["detected_flux_err_min"]>0, data["flux_ratio_sq_sum"], ((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["flux_ratio_sq_sum"]))) * 2.0) ) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_median"] > -1, data["detected_mjd_diff"], data["detected_flux_std"] )) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_0__attr__abs__y"]>0, data["2__kurtosis_x"], (((data["flux_by_flux_ratio_sq_sum"]) > (data["4__fft_coefficient__coeff_0__attr__abs__y"]))*1.) )) +
                0.100000*np.tanh(((((((np.where(data["flux_err_median"]>0, data["3__fft_coefficient__coeff_1__attr__abs__y"], np.maximum(((((data["flux_median"]) * 2.0))), ((data["detected_flux_err_min"]))) )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"]<0, data["mjd_diff"], np.where(data["mjd_diff"]>0, data["distmod"], data["distmod"] ) )) +
                0.100000*np.tanh((((((data["flux_median"]) < (np.where(data["3__skewness_x"]<0, data["detected_mjd_diff"], data["detected_flux_ratio_sq_sum"] )))*1.)) * (np.where(data["1__kurtosis_y"]>0, data["flux_median"], data["1__kurtosis_y"] )))) +
                0.100000*np.tanh(((((data["4__fft_coefficient__coeff_0__attr__abs__x"]) * (data["detected_flux_ratio_sq_skew"]))) * 2.0)) +
                0.100000*np.tanh(((((np.maximum(((data["2__kurtosis_x"])), ((data["flux_d1_pb3"])))) + (data["3__skewness_x"]))) * (((data["2__kurtosis_x"]) * (data["flux_d1_pb2"]))))) +
                0.100000*np.tanh(((data["distmod"]) + ((((((data["distmod"]) + (data["distmod"]))) + (data["hostgal_photoz_err"]))/2.0)))) +
                0.100000*np.tanh(np.where(data["detected_flux_median"]>0, data["4__kurtosis_x"], np.where(data["0__kurtosis_x"]>0, data["detected_flux_ratio_sq_sum"], data["4__kurtosis_x"] ) )) +
                0.100000*np.tanh((((((((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) > (np.where(data["flux_d1_pb0"]>0, data["detected_flux_std"], data["0__fft_coefficient__coeff_0__attr__abs__x"] )))*1.)) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh((((data["2__fft_coefficient__coeff_0__attr__abs__x"]) > (data["hostgal_photoz"]))*1.)) +
                0.100000*np.tanh(np.where(data["flux_diff"]>0, data["detected_flux_err_min"], data["1__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"]<0, data["2__fft_coefficient__coeff_0__attr__abs__x"], data["4__skewness_x"] )) +
                0.100000*np.tanh((((data["flux_d0_pb1"]) > (data["detected_flux_dif3"]))*1.)) +
                0.100000*np.tanh(np.where(data["flux_err_skew"]>0, np.where(((data["flux_err_skew"]) * 2.0)<0, data["2__kurtosis_x"], data["flux_median"] ), data["flux_ratio_sq_sum"] )) +
                0.100000*np.tanh(((((data["3__kurtosis_x"]) * (data["detected_flux_err_skew"]))) * 2.0)) +
                0.100000*np.tanh(((np.maximum(((data["detected_flux_median"])), ((data["1__skewness_y"])))) - (data["1__kurtosis_x"]))) +
                0.100000*np.tanh((((data["flux_d0_pb2"]) > (np.where(np.maximum(((np.minimum(((data["flux_d0_pb2"])), ((data["detected_flux_dif3"]))))), ((data["flux_d0_pb2"]))) > -1, data["4__fft_coefficient__coeff_1__attr__abs__x"], data["flux_d0_pb2"] )))*1.)) +
                0.100000*np.tanh(np.where(data["4__skewness_x"] > -1, np.where(data["4__skewness_x"]<0, data["hostgal_photoz_err"], data["1__skewness_y"] ), data["hostgal_photoz_err"] )) +
                0.100000*np.tanh(((np.where(np.where(data["detected_mjd_diff"]>0, data["flux_d0_pb3"], data["flux_d1_pb0"] )>0, data["distmod"], data["flux_d0_pb1"] )) * 2.0)) +
                0.100000*np.tanh(((np.where(((data["detected_flux_err_median"]) * 2.0)<0, data["flux_d1_pb0"], np.where(data["flux_d1_pb0"]>0, data["distmod"], data["flux_d1_pb5"] ) )) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_median"]) * 2.0)) + (((data["flux_median"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((((np.maximum(((data["detected_mjd_diff"])), ((data["flux_err_skew"])))) + (((data["flux_d1_pb0"]) + (((data["1__kurtosis_y"]) + (data["detected_mjd_diff"]))))))) + (data["flux_d1_pb0"]))))

    def GP_class_16(self,data):
        return (-1.007018 +
                0.100000*np.tanh((-1.0*((((data["flux_skew"]) + (((((data["flux_ratio_sq_sum"]) + (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) + (((((data["flux_skew"]) * 2.0)) + (data["flux_skew"])))))))))) +
                0.100000*np.tanh((((((-1.0*((((data["flux_skew"]) * 2.0))))) - (data["flux_skew"]))) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh((-1.0*((((((((((data["flux_skew"]) * 2.0)) + (data["4__skewness_x"]))) + ((1.0)))) + (data["5__fft_coefficient__coeff_0__attr__abs__y"])))))) +
                0.100000*np.tanh((-1.0*((((1.0) + (((np.maximum(((data["flux_skew"])), ((data["2__skewness_x"])))) + (((data["flux_skew"]) + (data["5__fft_coefficient__coeff_1__attr__abs__y"])))))))))) +
                0.100000*np.tanh((((-1.0*((((data["2__skewness_x"]) * 2.0))))) - (((((data["4__skewness_x"]) + (data["4__skewness_x"]))) + (3.0))))) +
                0.100000*np.tanh((-1.0*((np.where(np.where(np.where(2.718282 > -1, data["3__skewness_x"], data["distmod"] ) > -1, 2.718282, data["distmod"] ) > -1, 3.0, data["2__skewness_x"] ))))) +
                0.100000*np.tanh((-1.0*((((((data["flux_by_flux_ratio_sq_skew"]) * 2.0)) + (((data["detected_flux_mean"]) + (((data["flux_by_flux_ratio_sq_skew"]) + (((2.0) + (data["flux_ratio_sq_sum"])))))))))))) +
                0.100000*np.tanh((-1.0*((((data["flux_skew"]) + (data["detected_flux_by_flux_ratio_sq_skew"])))))) +
                0.100000*np.tanh(((((-2.0) - (data["2__skewness_x"]))) - (((data["flux_by_flux_ratio_sq_skew"]) - (((((-2.0) + (data["2__skewness_x"]))) - (data["2__skewness_x"]))))))) +
                0.100000*np.tanh(((np.where(((data["flux_d0_pb2"]) + (((data["3__skewness_x"]) * 2.0))) > -1, -2.0, -2.0 )) - (data["3__skewness_x"]))) +
                0.100000*np.tanh(((((-2.0) - (((((data["2__skewness_x"]) + (data["2__skewness_x"]))) + (((data["flux_by_flux_ratio_sq_skew"]) + (data["flux_by_flux_ratio_sq_skew"]))))))) - (2.718282))) +
                0.100000*np.tanh((-1.0*((((((data["flux_skew"]) + (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) + (((data["flux_by_flux_ratio_sq_skew"]) + (data["3__skewness_x"])))))))) +
                0.100000*np.tanh(((np.minimum(((((data["flux_skew"]) - (data["flux_skew"])))), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))) - (data["flux_skew"]))) +
                0.100000*np.tanh(np.where(-2.0 > -1, -2.0, np.where(((-2.0) - (data["2__skewness_x"]))<0, -2.0, ((-2.0) - (data["2__skewness_x"])) ) )) +
                0.100000*np.tanh(((((-3.0) - (((data["flux_skew"]) + (data["flux_skew"]))))) + (((data["flux_skew"]) - (data["flux_skew"]))))) +
                0.100000*np.tanh(np.where(data["3__skewness_x"] > -1, -2.0, (((((((-2.0) > (-2.0))*1.)) - (data["2__skewness_x"]))) - (data["flux_skew"])) )) +
                0.100000*np.tanh(((((-2.0) - (np.where(-2.0 > -1, ((-2.0) - (data["2__skewness_x"])), data["2__skewness_x"] )))) * 2.0)) +
                0.100000*np.tanh((-1.0*((((data["flux_skew"]) + (np.where(data["flux_by_flux_ratio_sq_skew"] > -1, 2.718282, data["5__fft_coefficient__coeff_1__attr__abs__x"] ))))))) +
                0.100000*np.tanh((-1.0*((((((data["detected_flux_ratio_sq_sum"]) + (((data["2__skewness_x"]) / 2.0)))) + (data["4__skewness_x"])))))) +
                0.100000*np.tanh(((((-3.0) - (data["2__skewness_x"]))) - (data["2__skewness_x"]))) +
                0.100000*np.tanh(((((data["flux_err_min"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) + (((data["flux_err_min"]) + (((data["flux_median"]) + (data["flux_err_min"]))))))) +
                0.100000*np.tanh(((((data["flux_err_min"]) - (data["detected_flux_err_std"]))) * 2.0)) +
                0.100000*np.tanh(((((-3.0) - (data["4__skewness_x"]))) - (data["flux_skew"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_max"] > -1, ((data["flux_err_min"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((data["flux_err_min"]) - (((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0))) )) +
                0.100000*np.tanh(((np.where(np.minimum(((-2.0)), ((data["flux_median"])))<0, ((-2.0) - (data["1__skewness_x"])), (-1.0*((data["4__skewness_x"]))) )) - (data["4__skewness_x"]))) +
                0.100000*np.tanh(((np.minimum(((((data["detected_flux_err_min"]) - (((np.maximum(((data["3__skewness_x"])), ((data["flux_median"])))) + (data["1__skewness_x"])))))), ((-1.0)))) - (data["1__skewness_x"]))) +
                0.100000*np.tanh(((((data["flux_err_min"]) + (data["flux_median"]))) - (data["flux_skew"]))) +
                0.100000*np.tanh(((((((data["flux_err_min"]) + (data["flux_err_min"]))) + (data["flux_err_min"]))) + (data["flux_err_min"]))) +
                0.100000*np.tanh((-1.0*((((data["2__skewness_x"]) - (np.minimum(((((-2.0) - (data["detected_flux_by_flux_ratio_sq_skew"])))), ((-2.0))))))))) +
                0.100000*np.tanh(((((np.minimum(((data["flux_median"])), ((data["detected_mjd_diff"])))) - (data["flux_skew"]))) - (data["1__skewness_x"]))) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz"] > -1, -3.0, ((((data["4__kurtosis_x"]) - (data["2__skewness_x"]))) - (data["2__skewness_x"])) )) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((((((data["flux_err_min"]) + (((data["flux_err_min"]) - (data["flux_err_min"]))))) + (data["flux_d1_pb0"]))) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((((((((((data["flux_err_min"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((-2.0)), ((((data["2__skewness_x"]) - (data["3__skewness_x"])))))) - (data["3__skewness_x"]))) +
                0.100000*np.tanh(((((data["flux_median"]) - (((data["4__skewness_x"]) - (((data["flux_skew"]) - (data["2__skewness_x"]))))))) - (data["4__skewness_x"]))) +
                0.100000*np.tanh(((np.minimum(((-1.0)), ((data["flux_ratio_sq_skew"])))) - (data["flux_by_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(np.minimum(((((data["detected_flux_err_min"]) + (data["flux_d0_pb0"])))), ((np.where(data["5__kurtosis_y"]<0, data["flux_d0_pb0"], data["detected_flux_err_min"] ))))) +
                0.100000*np.tanh(np.where(((data["flux_err_min"]) / 2.0)>0, data["detected_flux_err_min"], ((data["detected_mjd_diff"]) - (data["flux_err_min"])) )) +
                0.100000*np.tanh(((np.minimum(((data["flux_median"])), ((((data["flux_median"]) - (data["detected_flux_min"])))))) - (data["detected_flux_min"]))) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, ((np.minimum(((data["3__skewness_x"])), ((((-3.0) - (data["3__skewness_x"])))))) * 2.0), data["detected_flux_err_min"] )) +
                0.100000*np.tanh(((data["flux_err_min"]) + (np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["detected_flux_err_min"], (-1.0*((data["flux_min"]))) )))) +
                0.100000*np.tanh((((((((((data["flux_skew"]) - (data["flux_skew"]))) - (data["flux_skew"]))) > (data["1__skewness_x"]))*1.)) - (data["flux_skew"]))) +
                0.100000*np.tanh(np.minimum(((data["flux_median"])), ((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_median"])))))) +
                0.100000*np.tanh(((((np.minimum(((np.minimum(((((data["flux_err_min"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["flux_err_min"]))))), ((data["flux_err_min"])))) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) +
                0.100000*np.tanh(((((((np.minimum(((data["flux_err_min"])), ((data["detected_mjd_diff"])))) + (data["flux_err_min"]))) * 2.0)) + (np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["flux_err_min"])))))) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, data["flux_median"], np.where(((data["flux_median"]) - (data["detected_flux_min"]))>0, ((data["flux_median"]) - (data["detected_flux_min"])), -3.0 ) )) +
                0.100000*np.tanh(np.minimum(((data["flux_err_min"])), ((np.minimum(((data["detected_flux_diff"])), ((data["flux_err_min"]))))))) +
                0.100000*np.tanh(((((-1.0) - (np.where(data["1__skewness_x"]>0, (-1.0*((((-1.0) - (data["1__fft_coefficient__coeff_0__attr__abs__x"]))))), data["detected_flux_skew"] )))) - (data["4__skewness_x"]))) +
                0.100000*np.tanh(((((np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"] > -1, ((data["0__kurtosis_y"]) + (data["detected_mjd_diff"])), data["flux_ratio_sq_skew"] )) + (data["flux_err_min"]))) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(np.where(data["2__skewness_x"] > -1, data["flux_d0_pb5"], (-1.0*((data["2__skewness_x"]))) )) +
                0.100000*np.tanh(np.where(data["0__kurtosis_x"]<0, ((data["flux_median"]) + (((np.minimum(((data["detected_flux_ratio_sq_skew"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))) + (0.0)))), data["flux_d0_pb0"] )) +
                0.100000*np.tanh(((np.where(((data["2__fft_coefficient__coeff_0__attr__abs__x"]) - (data["detected_flux_skew"])) > -1, -2.0, data["flux_ratio_sq_skew"] )) - (data["distmod"]))) +
                0.100000*np.tanh(np.minimum(((((-3.0) + (((-3.0) * (((data["2__fft_coefficient__coeff_0__attr__abs__y"]) - (data["5__skewness_x"])))))))), ((2.718282)))) +
                0.100000*np.tanh(((((data["5__kurtosis_y"]) - ((((((data["1__skewness_x"]) - (data["flux_d0_pb5"]))) > (data["2__fft_coefficient__coeff_0__attr__abs__y"]))*1.)))) - (data["2__skewness_x"]))) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + (((data["flux_err_skew"]) + (((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (data["flux_ratio_sq_skew"]))))))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_d0_pb0"]))) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + (((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (data["1__fft_coefficient__coeff_0__attr__abs__x"]))))) +
                0.100000*np.tanh(((np.minimum(((data["flux_ratio_sq_skew"])), ((data["0__kurtosis_x"])))) + (np.minimum(((data["detected_flux_std"])), ((data["flux_ratio_sq_sum"])))))) +
                0.100000*np.tanh(np.where(data["detected_flux_mean"]>0, -3.0, np.where(-3.0>0, data["detected_flux_mean"], data["detected_flux_std"] ) )) +
                0.100000*np.tanh(np.where((((np.where(-2.0 > -1, ((-3.0) - (data["4__fft_coefficient__coeff_0__attr__abs__x"])), data["1__fft_coefficient__coeff_0__attr__abs__y"] )) > (data["flux_err_min"]))*1.) > -1, data["flux_err_min"], data["mjd_diff"] )) +
                0.100000*np.tanh(((-2.0) - (data["3__skewness_x"]))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + ((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["flux_d0_pb0"]) + (np.where(data["flux_err_min"] > -1, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["flux_d0_pb3"] )))))/2.0)))) +
                0.100000*np.tanh((((((((data["detected_mjd_diff"]) + (((data["detected_mjd_diff"]) - (data["3__skewness_x"]))))) + (((data["flux_median"]) - (data["flux_median"]))))) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) +
                0.100000*np.tanh(((data["flux_median"]) + ((((((data["flux_median"]) + (data["detected_mjd_diff"]))/2.0)) - (data["2__skewness_x"]))))) +
                0.100000*np.tanh(((((((data["flux_d0_pb0"]) + (data["flux_d0_pb0"]))) - (((data["flux_dif2"]) - (data["2__skewness_x"]))))) * 2.0)) +
                0.100000*np.tanh(((((-3.0) + (((np.tanh(((-1.0*((data["detected_flux_by_flux_ratio_sq_sum"])))))) * 2.0)))) / 2.0)) +
                0.100000*np.tanh(((np.where(data["detected_flux_min"] > -1, ((data["flux_d0_pb5"]) - (data["flux_mean"])), data["flux_mean"] )) - (((((data["detected_flux_min"]) * 2.0)) - (data["flux_mean"]))))) +
                0.100000*np.tanh(((np.where(((data["flux_err_min"]) / 2.0)>0, data["flux_err_min"], ((data["2__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0) )) + (data["flux_d0_pb0"]))) +
                0.100000*np.tanh(((np.minimum(((data["3__fft_coefficient__coeff_0__attr__abs__x"])), ((data["detected_mjd_diff"])))) + ((((-1.0*((data["0__fft_coefficient__coeff_1__attr__abs__x"])))) - (data["4__kurtosis_y"]))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["detected_flux_err_min"])), ((data["flux_err_min"]))))), ((data["detected_flux_std"])))) +
                0.100000*np.tanh(((np.where(data["distmod"]>0, data["flux_median"], ((((data["flux_median"]) - (data["detected_flux_err_max"]))) * 2.0) )) - (data["detected_flux_min"]))) +
                0.100000*np.tanh(np.tanh((((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(((((((data["flux_mean"]) > (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) / 2.0)))*1.)) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)) +
                0.100000*np.tanh(np.where(data["1__skewness_x"]>0, -3.0, np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]>0, data["1__fft_coefficient__coeff_1__attr__abs__x"], -3.0 ) )) +
                0.100000*np.tanh((((((-1.0*(((((data["flux_d1_pb5"]) < (data["detected_flux_median"]))*1.))))) - (data["detected_flux_w_mean"]))) - (data["detected_flux_median"]))) +
                0.100000*np.tanh(((np.where((-1.0*((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (data["flux_ratio_sq_sum"])))))>0, data["flux_ratio_sq_skew"], data["0__kurtosis_x"] )) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((data["flux_err_min"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.tanh((data["flux_ratio_sq_skew"]))) + ((((data["detected_mjd_diff"]) + (data["2__kurtosis_y"]))/2.0)))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["detected_mjd_diff"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh((((((((((((data["flux_max"]) + (data["flux_ratio_sq_skew"]))/2.0)) * 2.0)) + (data["detected_mjd_diff"]))/2.0)) + (data["flux_ratio_sq_skew"]))/2.0)) +
                0.100000*np.tanh(((data["detected_flux_ratio_sq_skew"]) - (data["1__skewness_x"]))) +
                0.100000*np.tanh(((((((data["flux_d0_pb0"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) + (((data["detected_flux_std"]) + (data["detected_flux_std"]))))) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__x"])), ((np.minimum(((np.where(data["flux_ratio_sq_skew"]<0, data["detected_mjd_diff"], data["flux_err_min"] ))), ((data["flux_max"]))))))) +
                0.100000*np.tanh(np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), (((((((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_max"]))) + (data["5__fft_coefficient__coeff_0__attr__abs__x"]))/2.0))))) +
                0.100000*np.tanh(np.where(((data["flux_dif3"]) - (data["1__skewness_x"]))>0, (-1.0*((data["detected_flux_median"]))), data["flux_d0_pb0"] )) +
                0.100000*np.tanh(((np.where(data["flux_ratio_sq_skew"] > -1, ((data["detected_mjd_diff"]) - (data["detected_flux_err_std"])), data["flux_err_min"] )) + (data["flux_err_min"]))) +
                0.100000*np.tanh((((data["flux_ratio_sq_skew"]) + ((((data["flux_d0_pb0"]) + (data["flux_d0_pb0"]))/2.0)))/2.0)) +
                0.100000*np.tanh(((data["mjd_size"]) + (np.where(((data["detected_flux_err_max"]) + (((data["flux_ratio_sq_skew"]) + (data["detected_flux_err_min"])))) > -1, data["detected_flux_err_min"], data["flux_skew"] )))) +
                0.100000*np.tanh((((data["4__kurtosis_x"]) + ((((((data["5__skewness_y"]) + (data["flux_err_min"]))) + (data["detected_flux_max"]))/2.0)))/2.0)) +
                0.100000*np.tanh(np.minimum((((((((data["flux_by_flux_ratio_sq_skew"]) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)) + (data["2__fft_coefficient__coeff_0__attr__abs__x"])))), ((((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (data["detected_flux_err_min"]))) + (data["2__fft_coefficient__coeff_0__attr__abs__y"])))))) +
                0.100000*np.tanh(np.where(data["1__skewness_x"]<0, data["flux_err_min"], ((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((data["4__skewness_y"])))) - (data["1__skewness_x"])) )) +
                0.100000*np.tanh(np.minimum(((data["2__fft_coefficient__coeff_0__attr__abs__x"])), ((np.where(data["flux_diff"]>0, data["2__fft_coefficient__coeff_0__attr__abs__x"], data["detected_mjd_diff"] ))))) +
                0.100000*np.tanh((((((-1.0*((((data["flux_d1_pb5"]) - (data["detected_flux_err_mean"])))))) - ((-1.0*((data["5__fft_coefficient__coeff_1__attr__abs__x"])))))) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh((((-1.0*((((((np.maximum(((data["flux_d0_pb1"])), ((data["flux_d0_pb0"])))) - (data["ddf"]))) - (data["1__kurtosis_x"])))))) / 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_err_min"] > -1, data["0__kurtosis_y"], ((data["2__kurtosis_y"]) + (0.0)) )) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_err_min"])), ((((np.minimum(((((data["detected_flux_diff"]) - ((((((data["flux_err_min"]) + (data["flux_err_min"]))/2.0)) / 2.0))))), ((data["detected_flux_err_min"])))) * 2.0))))) +
                0.100000*np.tanh(np.where(data["detected_flux_dif2"] > -1, np.where(data["detected_flux_dif2"]>0, data["2__kurtosis_y"], data["flux_d1_pb0"] ), data["detected_mjd_diff"] )) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) + (data["flux_d0_pb0"]))) +
                0.100000*np.tanh(((((np.where(data["detected_mjd_diff"] > -1, -3.0, ((data["2__skewness_y"]) * 2.0) )) + (data["3__skewness_y"]))) + (data["flux_err_min"]))) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + (np.where(data["detected_flux_diff"] > -1, np.where(np.minimum(((data["3__fft_coefficient__coeff_1__attr__abs__y"])), ((data["detected_mjd_diff"]))) > -1, data["1__fft_coefficient__coeff_0__attr__abs__x"], data["detected_flux_ratio_sq_skew"] ), data["flux_median"] )))) +
                0.100000*np.tanh((((((data["mjd_diff"]) - (np.minimum(((data["detected_flux_err_skew"])), ((data["detected_flux_err_mean"])))))) + (data["flux_ratio_sq_skew"]))/2.0)) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, data["2__fft_coefficient__coeff_1__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh((-1.0*((np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"] > -1, np.maximum(((((data["2__skewness_x"]) - (data["1__kurtosis_x"])))), ((data["2__skewness_x"]))), data["ddf"] ))))) +
                0.100000*np.tanh((((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) + (((data["flux_err_min"]) + (data["detected_mjd_diff"]))))/2.0)) +
                0.100000*np.tanh(((((np.minimum(((data["flux_err_min"])), ((data["flux_ratio_sq_skew"])))) - (-2.0))) + (((data["flux_err_min"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), ((data["1__fft_coefficient__coeff_0__attr__abs__y"])))) +
                0.100000*np.tanh(np.where(data["flux_dif3"] > -1, ((data["flux_dif3"]) - (data["flux_err_std"])), ((data["4__kurtosis_y"]) * 2.0) )) +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((data["flux_dif3"])), ((data["0__kurtosis_x"]))))), ((data["flux_dif3"]))))), ((data["flux_dif3"])))) +
                0.100000*np.tanh(((np.where(data["flux_err_min"]>0, data["5__kurtosis_x"], np.minimum(((data["detected_flux_err_min"])), ((data["flux_dif3"]))) )) + (((data["flux_dif3"]) + (data["detected_mjd_diff"]))))) +
                0.100000*np.tanh(((((data["detected_flux_err_min"]) + (((data["flux_d0_pb5"]) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))))) + ((((data["detected_flux_max"]) + (data["detected_flux_err_min"]))/2.0)))) +
                0.100000*np.tanh((((((((data["detected_flux_err_mean"]) + (data["flux_d1_pb5"]))/2.0)) - (data["2__skewness_x"]))) + ((-1.0*((data["3__fft_coefficient__coeff_1__attr__abs__y"])))))) +
                0.100000*np.tanh(np.where((-1.0*((data["2__fft_coefficient__coeff_0__attr__abs__y"]))) > -1, data["4__skewness_y"], data["2__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(((np.where(np.where((-1.0*((data["detected_flux_std"])))>0, data["4__skewness_y"], data["flux_std"] )>0, data["detected_flux_std"], data["4__skewness_y"] )) / 2.0)) +
                0.100000*np.tanh((((((data["flux_ratio_sq_skew"]) - (np.minimum(((data["hostgal_photoz_err"])), ((data["4__fft_coefficient__coeff_1__attr__abs__x"])))))) + (np.tanh((((data["0__kurtosis_x"]) / 2.0)))))/2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_std"]>0, data["flux_err_skew"], ((data["flux_skew"]) * 2.0) )) +
                0.100000*np.tanh(((data["4__skewness_y"]) + ((((data["4__skewness_y"]) + (data["detected_flux_std"]))/2.0)))) +
                0.100000*np.tanh((((data["0__kurtosis_x"]) + (data["detected_mjd_diff"]))/2.0)) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["2__fft_coefficient__coeff_0__attr__abs__y"])), (((((np.maximum(((data["flux_dif3"])), ((data["ddf"])))) + (((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_d0_pb0"]))))/2.0)))))), ((data["2__fft_coefficient__coeff_0__attr__abs__y"])))) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_1__attr__abs__y"] > -1, ((((np.minimum(((data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((data["flux_d1_pb2"])))) - (data["flux_err_std"]))) * (data["flux_max"])), data["3__skewness_x"] )) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) * (data["detected_flux_err_min"]))))

    def GP_class_42(self,data):
        return (-0.859449 +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((data["hostgal_photoz"])), ((-2.0))))), ((-3.0))))), ((-3.0)))) +
                0.100000*np.tanh(np.minimum(((-2.0)), ((-2.0)))) +
                0.100000*np.tanh(np.minimum(((-3.0)), ((np.minimum(((np.minimum(((data["flux_mean"])), ((np.minimum(((data["detected_flux_min"])), ((-2.0)))))))), ((-3.0))))))) +
                0.100000*np.tanh(((np.minimum(((((np.minimum(((data["distmod"])), ((data["detected_flux_min"])))) + (data["flux_d1_pb4"])))), ((data["flux_d1_pb3"])))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((np.minimum(((((data["distmod"]) * 2.0))), ((data["detected_flux_min"]))))), ((-3.0)))) +
                0.100000*np.tanh(np.minimum(((data["3__skewness_x"])), ((((np.minimum(((data["flux_min"])), ((data["flux_min"])))) * 2.0))))) +
                0.100000*np.tanh(np.minimum(((((data["distmod"]) / 2.0))), ((np.tanh((-3.0)))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["hostgal_photoz_err"])), ((data["2__skewness_x"]))))), ((data["flux_d0_pb5"])))) +
                0.100000*np.tanh(np.minimum(((data["2__skewness_x"])), ((np.minimum(((data["flux_min"])), ((np.minimum(((data["detected_flux_by_flux_ratio_sq_sum"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))))))))) +
                0.100000*np.tanh(np.minimum(((np.where(data["flux_min"]>0, data["detected_flux_min"], data["flux_min"] ))), ((data["detected_flux_min"])))) +
                0.100000*np.tanh(np.minimum(((data["flux_min"])), ((data["flux_min"])))) +
                0.100000*np.tanh(np.minimum(((data["flux_d1_pb2"])), ((np.minimum(((-2.0)), ((data["flux_dif2"]))))))) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (np.tanh((data["3__skewness_x"]))))) + (np.minimum(((data["detected_flux_min"])), ((((((data["distmod"]) + (data["flux_d0_pb4"]))) * 2.0))))))) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_min"])), ((data["flux_min"])))) +
                0.100000*np.tanh(np.minimum(((data["distmod"])), ((np.minimum(((data["flux_dif2"])), ((np.where(np.minimum(((data["detected_flux_by_flux_ratio_sq_skew"])), ((data["flux_by_flux_ratio_sq_skew"])))<0, data["flux_diff"], data["distmod"] )))))))) +
                0.100000*np.tanh(((((((((data["detected_flux_min"]) - (data["flux_ratio_sq_skew"]))) - (data["hostgal_photoz_err"]))) - (data["detected_flux_std"]))) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((np.minimum(((data["distmod"])), ((data["flux_d0_pb4"])))) + (((data["detected_flux_min"]) + (data["detected_flux_min"]))))) +
                0.100000*np.tanh(((((data["flux_min"]) + (np.minimum(((data["detected_flux_min"])), ((data["detected_flux_min"])))))) + (data["detected_flux_min"]))) +
                0.100000*np.tanh(np.minimum(((((data["distmod"]) - (np.minimum(((data["distmod"])), ((data["distmod"]))))))), ((np.minimum(((data["mjd_size"])), ((np.minimum(((data["distmod"])), ((data["distmod"])))))))))) +
                0.100000*np.tanh(((np.minimum(((data["distmod"])), ((data["distmod"])))) + (((data["distmod"]) + (data["1__kurtosis_x"]))))) +
                0.100000*np.tanh(((((data["1__kurtosis_x"]) + (data["flux_d0_pb5"]))) + (np.minimum(((data["1__kurtosis_x"])), ((data["detected_flux_min"])))))) +
                0.100000*np.tanh(((np.where(np.maximum(((data["detected_flux_std"])), (((((data["detected_flux_min"]) > (data["flux_d0_pb4"]))*1.)))) > -1, data["detected_flux_min"], data["detected_flux_ratio_sq_skew"] )) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((((data["5__kurtosis_y"]) + (data["5__kurtosis_y"])))), ((data["detected_flux_min"]))))), ((data["flux_min"])))) +
                0.100000*np.tanh(((data["detected_flux_min"]) + (data["distmod"]))) +
                0.100000*np.tanh(((np.minimum(((((data["flux_d1_pb2"]) / 2.0))), ((((data["detected_flux_min"]) + (((data["hostgal_photoz_err"]) + (data["detected_flux_min"])))))))) + (data["flux_d0_pb5"]))) +
                0.100000*np.tanh(((((((((((data["detected_flux_std"]) + (data["flux_diff"]))) / 2.0)) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) / 2.0)) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((data["detected_flux_min"]) + (data["detected_flux_min"]))) +
                0.100000*np.tanh(((((data["hostgal_photoz_err"]) + (((((data["detected_flux_min"]) + (((data["flux_min"]) + (data["hostgal_photoz_err"]))))) * 2.0)))) + (data["1__kurtosis_x"]))) +
                0.100000*np.tanh(((((data["flux_diff"]) - (data["flux_diff"]))) - (data["flux_diff"]))) +
                0.100000*np.tanh(((data["0__kurtosis_x"]) - (np.maximum(((data["5__skewness_x"])), ((data["flux_diff"])))))) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (data["flux_d0_pb5"]))) * 2.0)) +
                0.100000*np.tanh(((((data["flux_mean"]) + (((data["detected_flux_dif3"]) - (data["flux_max"]))))) - (data["flux_max"]))) +
                0.100000*np.tanh((((data["detected_flux_min"]) + (((data["detected_flux_min"]) + (data["4__fft_coefficient__coeff_0__attr__abs__y"]))))/2.0)) +
                0.100000*np.tanh((((-1.0*((((data["detected_flux_std"]) - (((data["0__kurtosis_x"]) - (np.where(data["flux_mean"]>0, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["detected_flux_std"] ))))))))) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"] > -1, (((data["0__kurtosis_y"]) + (np.tanh((data["5__fft_coefficient__coeff_1__attr__abs__x"]))))/2.0), ((((data["flux_d0_pb4"]) + (data["detected_mean"]))) + (data["flux_d0_pb4"])) )) +
                0.100000*np.tanh((((((data["4__skewness_x"]) + (((((data["detected_flux_max"]) + (((data["detected_flux_min"]) * 2.0)))) + (data["detected_flux_min"]))))/2.0)) + (data["1__kurtosis_x"]))) +
                0.100000*np.tanh(((data["hostgal_photoz_err"]) + (((data["1__kurtosis_x"]) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh((((((((1.0) > (data["detected_flux_std"]))*1.)) - (data["detected_flux_std"]))) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((data["flux_d0_pb5"]) + (np.maximum(((data["5__skewness_x"])), ((data["flux_d0_pb5"])))))) +
                0.100000*np.tanh(((((((((((data["detected_flux_ratio_sq_skew"]) - (data["detected_mean"]))) - (data["3__kurtosis_x"]))) - (data["detected_flux_err_std"]))) - (data["3__kurtosis_x"]))) - (data["3__kurtosis_x"]))) +
                0.100000*np.tanh(((data["flux_d0_pb4"]) + (np.tanh((data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh((((((data["5__fft_coefficient__coeff_0__attr__abs__x"]) < (data["1__skewness_y"]))*1.)) - (data["detected_flux_skew"]))) +
                0.100000*np.tanh(np.tanh((((data["4__kurtosis_y"]) + (((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))))) +
                0.100000*np.tanh(np.tanh((((data["detected_flux_min"]) + (((data["detected_flux_min"]) + (np.maximum(((((data["flux_mean"]) - (data["flux_dif3"])))), ((data["5__skewness_x"])))))))))) +
                0.100000*np.tanh((-1.0*((((data["3__kurtosis_x"]) + (data["distmod"])))))) +
                0.100000*np.tanh(((data["hostgal_photoz_err"]) + (data["1__kurtosis_y"]))) +
                0.100000*np.tanh(((data["detected_flux_by_flux_ratio_sq_skew"]) + (data["1__kurtosis_x"]))) +
                0.100000*np.tanh((((data["0__kurtosis_x"]) + ((((((data["detected_flux_min"]) - (((data["detected_flux_min"]) + (data["3__kurtosis_y"]))))) + (data["distmod"]))/2.0)))/2.0)) +
                0.100000*np.tanh(((((((((np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__x"])), ((data["4__fft_coefficient__coeff_0__attr__abs__x"])))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["detected_flux_skew"]))) +
                0.100000*np.tanh(((np.where(data["detected_flux_by_flux_ratio_sq_skew"]<0, data["detected_flux_dif2"], (((data["detected_mjd_diff"]) > (data["detected_flux_dif2"]))*1.) )) * 2.0)) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_ratio_sq_skew"]))) + (((np.maximum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((((data["5__kurtosis_x"]) + (data["flux_by_flux_ratio_sq_skew"])))))) + (data["5__skewness_x"]))))) +
                0.100000*np.tanh(((((((((data["flux_d1_pb0"]) - (data["detected_flux_std"]))) + (data["hostgal_photoz"]))) - (np.tanh((data["flux_d1_pb0"]))))) * 2.0)) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_min"]))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["2__skewness_x"]) * (data["detected_flux_min"]))) + (data["detected_flux_min"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_ratio_sq_skew"]>0, data["4__kurtosis_y"], data["flux_d1_pb0"] )) +
                0.100000*np.tanh((((data["3__fft_coefficient__coeff_0__attr__abs__x"]) + (data["detected_flux_min"]))/2.0)) +
                0.100000*np.tanh(np.where(data["flux_diff"]<0, np.maximum(((np.where(data["flux_diff"]<0, data["detected_flux_ratio_sq_skew"], data["5__kurtosis_y"] ))), ((data["detected_flux_ratio_sq_skew"]))), ((data["5__kurtosis_y"]) + (data["hostgal_photoz_err"])) )) +
                0.100000*np.tanh(np.where(data["1__kurtosis_x"]>0, data["1__kurtosis_x"], ((np.where(data["flux_ratio_sq_skew"] > -1, data["1__kurtosis_x"], (-1.0*((data["1__kurtosis_x"]))) )) - (data["1__fft_coefficient__coeff_1__attr__abs__x"])) )) +
                0.100000*np.tanh((((np.where(data["flux_d0_pb4"]<0, data["detected_flux_dif2"], data["detected_mjd_diff"] )) > (data["detected_flux_std"]))*1.)) +
                0.100000*np.tanh(((data["detected_mjd_size"]) + (((data["detected_flux_min"]) + (data["flux_diff"]))))) +
                0.100000*np.tanh(((data["detected_flux_min"]) + (data["mjd_diff"]))) +
                0.100000*np.tanh((((data["flux_max"]) < (np.maximum((((((((data["detected_flux_std"]) + (data["0__kurtosis_x"]))) < (data["3__fft_coefficient__coeff_0__attr__abs__y"]))*1.))), ((data["3__kurtosis_x"])))))*1.)) +
                0.100000*np.tanh(((data["3__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["flux_d1_pb0"]) + (data["distmod"]))))) +
                0.100000*np.tanh(np.where(((data["detected_mjd_diff"]) - (data["detected_flux_w_mean"]))>0, np.where((((data["hostgal_photoz"]) > (data["hostgal_photoz"]))*1.)<0, data["flux_max"], data["detected_flux_w_mean"] ), data["hostgal_photoz"] )) +
                0.100000*np.tanh(((data["flux_d1_pb1"]) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((data["3__kurtosis_y"]) + (((((data["flux_by_flux_ratio_sq_skew"]) + (data["flux_d1_pb5"]))) + (data["2__kurtosis_x"]))))) +
                0.100000*np.tanh(np.where(((data["detected_flux_std"]) - (data["detected_flux_std"]))>0, data["detected_mjd_diff"], ((data["detected_mjd_diff"]) - (data["detected_flux_std"])) )) +
                0.100000*np.tanh(np.where(np.where(data["2__skewness_x"]<0, data["2__skewness_x"], np.maximum(((data["2__skewness_x"])), ((data["flux_skew"]))) )<0, -3.0, (13.35520839691162109) )) +
                0.100000*np.tanh(np.where(data["flux_d0_pb0"]<0, np.where(data["flux_d0_pb0"]<0, data["flux_d0_pb0"], data["flux_d0_pb0"] ), ((data["flux_d0_pb1"]) - (data["3__fft_coefficient__coeff_1__attr__abs__y"])) )) +
                0.100000*np.tanh(((data["flux_mean"]) + (((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__y"])), ((data["flux_skew"])))) + (data["4__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(((np.maximum(((data["1__kurtosis_x"])), (((((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) > (2.0))*1.)) - (data["4__fft_coefficient__coeff_0__attr__abs__y"])))))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_d1_pb5"]))) +
                0.100000*np.tanh(((data["hostgal_photoz_err"]) + ((((data["flux_d1_pb0"]) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) / 2.0)))/2.0)))) +
                0.100000*np.tanh(((((((data["0__kurtosis_x"]) - (data["detected_mjd_diff"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) * (data["flux_d0_pb0"]))) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_0__attr__abs__y"]) + (np.maximum(((((np.maximum(((data["4__fft_coefficient__coeff_0__attr__abs__y"])), ((data["flux_d0_pb5"])))) + (data["detected_flux_by_flux_ratio_sq_sum"])))), ((data["4__fft_coefficient__coeff_0__attr__abs__y"])))))) +
                0.100000*np.tanh(np.where(np.tanh((data["distmod"]))>0, data["flux_ratio_sq_skew"], np.where(data["hostgal_photoz"]>0, np.where(data["3__kurtosis_y"]>0, data["detected_flux_min"], data["3__fft_coefficient__coeff_0__attr__abs__x"] ), data["detected_flux_diff"] ) )) +
                0.100000*np.tanh(np.where(((data["distmod"]) * 2.0) > -1, np.where(data["hostgal_photoz"]<0, data["hostgal_photoz"], ((data["3__kurtosis_y"]) * 2.0) ), ((np.tanh((data["flux_d0_pb0"]))) * 2.0) )) +
                0.100000*np.tanh((((np.where(data["3__skewness_x"]>0, data["detected_mjd_diff"], data["flux_d0_pb3"] )) > (data["detected_flux_std"]))*1.)) +
                0.100000*np.tanh((((((((((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) - (data["0__kurtosis_y"]))) * 2.0)) < (((data["flux_d1_pb3"]) + (((data["3__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0)))))*1.)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["3__skewness_x"]) + (((data["detected_mjd_diff"]) * 2.0)))) +
                0.100000*np.tanh(((np.where(data["4__kurtosis_x"]<0, data["4__fft_coefficient__coeff_0__attr__abs__y"], ((((np.where(((data["distmod"]) * 2.0) > -1, data["hostgal_photoz"], data["4__kurtosis_x"] )) * 2.0)) * 2.0) )) * 2.0)) +
                0.100000*np.tanh((((data["distmod"]) + ((((((((((data["detected_flux_min"]) + (data["distmod"]))/2.0)) + (data["distmod"]))/2.0)) + (data["detected_flux_min"]))/2.0)))/2.0)) +
                0.100000*np.tanh(np.where(data["flux_err_skew"]>0, ((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) * (data["flux_err_skew"]))) + (data["4__fft_coefficient__coeff_0__attr__abs__y"])), ((data["4__fft_coefficient__coeff_0__attr__abs__y"]) * (data["2__skewness_y"])) )) +
                0.100000*np.tanh(np.where((((data["flux_by_flux_ratio_sq_sum"]) + (data["2__kurtosis_y"]))/2.0)>0, data["detected_flux_min"], np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"]>0, data["3__kurtosis_y"], data["3__kurtosis_y"] ) )) +
                0.100000*np.tanh(((((data["flux_d1_pb0"]) - (((data["3__fft_coefficient__coeff_1__attr__abs__y"]) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))))) * (data["0__kurtosis_x"]))) +
                0.100000*np.tanh(((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) + ((((data["0__skewness_x"]) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)))) + ((((data["distmod"]) + (((data["3__fft_coefficient__coeff_0__attr__abs__y"]) + (data["0__skewness_x"]))))/2.0)))) +
                0.100000*np.tanh(np.where(data["flux_dif2"]>0, np.where(data["hostgal_photoz"]>0, data["detected_flux_ratio_sq_skew"], data["distmod"] ), data["mjd_diff"] )) +
                0.100000*np.tanh(np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["5__skewness_x"], ((data["detected_flux_dif3"]) + (data["1__skewness_y"])) )) +
                0.100000*np.tanh(np.where(data["2__skewness_y"]>0, data["flux_d0_pb4"], np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"]<0, data["flux_d0_pb0"], data["2__kurtosis_y"] ) )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb0"]>0, data["detected_flux_min"], data["distmod"] )) +
                0.100000*np.tanh(((((data["detected_flux_err_min"]) + (data["1__kurtosis_x"]))) * (data["flux_d1_pb1"]))) +
                0.100000*np.tanh(((((data["detected_flux_by_flux_ratio_sq_sum"]) + (data["flux_d1_pb3"]))) + (((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) + (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_w_mean"]))))))) +
                0.100000*np.tanh(np.where(data["detected_flux_min"] > -1, data["detected_flux_min"], data["detected_flux_min"] )) +
                0.100000*np.tanh(np.where(data["distmod"]>0, data["5__kurtosis_x"], data["0__fft_coefficient__coeff_0__attr__abs__x"] )) +
                0.100000*np.tanh(np.maximum((((((data["detected_mjd_diff"]) + (data["detected_flux_err_max"]))/2.0))), ((data["mjd_diff"])))) +
                0.100000*np.tanh(np.minimum(((((((data["flux_err_skew"]) - (np.tanh((data["mwebv"]))))) / 2.0))), ((data["flux_err_skew"])))) +
                0.100000*np.tanh(np.where(data["detected_flux_dif2"] > -1, ((((((data["detected_mjd_diff"]) > (np.where(data["detected_mjd_diff"] > -1, data["detected_flux_dif2"], data["detected_flux_dif2"] )))*1.)) > (data["detected_flux_dif2"]))*1.), data["detected_flux_dif2"] )) +
                0.100000*np.tanh(((np.where(data["detected_mean"]>0, data["flux_dif2"], data["distmod"] )) + (np.maximum(((data["distmod"])), ((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["5__fft_coefficient__coeff_1__attr__abs__x"])))))))) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["3__kurtosis_y"], np.where(data["flux_d0_pb0"]>0, np.where(data["flux_d0_pb1"]>0, data["3__kurtosis_y"], data["flux_d0_pb0"] ), data["5__fft_coefficient__coeff_1__attr__abs__x"] ) )) +
                0.100000*np.tanh((((data["detected_flux_w_mean"]) + (((data["detected_flux_err_std"]) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))))/2.0)) +
                0.100000*np.tanh(np.where(data["3__kurtosis_y"]>0, data["5__skewness_y"], (((data["5__skewness_y"]) > (((((((data["5__skewness_y"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) + (data["5__skewness_y"]))/2.0)))*1.) )) +
                0.100000*np.tanh((((((((((data["flux_d1_pb1"]) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)) + (data["distmod"]))/2.0)) +
                0.100000*np.tanh(np.where(data["flux_by_flux_ratio_sq_skew"]<0, (((((data["1__skewness_x"]) > (data["1__skewness_x"]))*1.)) / 2.0), data["flux_by_flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(np.where(data["detected_flux_skew"]<0, data["0__skewness_x"], np.where(data["hostgal_photoz"] > -1, ((data["hostgal_photoz"]) * 2.0), data["hostgal_photoz"] ) )) +
                0.100000*np.tanh((((((((data["flux_d1_pb3"]) - (data["0__kurtosis_y"]))) - (data["0__kurtosis_y"]))) + (((data["flux_d1_pb3"]) * 2.0)))/2.0)) +
                0.100000*np.tanh(np.where(data["flux_dif2"]<0, data["hostgal_photoz_err"], np.where(data["flux_dif2"] > -1, data["1__fft_coefficient__coeff_1__attr__abs__y"], np.where(data["flux_dif2"] > -1, data["detected_flux_err_median"], data["detected_flux_err_median"] ) ) )) +
                0.100000*np.tanh(np.maximum(((((data["1__kurtosis_x"]) * (data["1__kurtosis_x"])))), ((data["1__kurtosis_x"])))) +
                0.100000*np.tanh(np.maximum(((data["detected_flux_err_std"])), ((data["flux_d1_pb1"])))) +
                0.100000*np.tanh(np.tanh((((((((np.minimum(((((data["detected_mjd_diff"]) / 2.0))), ((data["3__skewness_y"])))) + (data["0__kurtosis_y"]))/2.0)) + (data["0__kurtosis_y"]))/2.0)))) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]>0, data["flux_d0_pb0"], np.where(data["flux_dif2"]>0, data["flux_dif2"], data["5__fft_coefficient__coeff_1__attr__abs__x"] ) )) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_0__attr__abs__y"]>0, np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"]<0, data["5__kurtosis_x"], data["4__fft_coefficient__coeff_0__attr__abs__x"] ), data["mjd_diff"] )) +
                0.100000*np.tanh((((((((data["5__kurtosis_y"]) * ((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)))) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) +
                0.100000*np.tanh((((data["distmod"]) > (data["detected_mjd_diff"]))*1.)) +
                0.100000*np.tanh(np.where(data["2__skewness_y"] > -1, np.where(((data["detected_mjd_diff"]) + (data["2__kurtosis_y"])) > -1, np.maximum(((data["detected_mjd_diff"])), ((data["1__fft_coefficient__coeff_0__attr__abs__x"]))), data["detected_flux_dif3"] ), data["1__fft_coefficient__coeff_0__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(data["2__kurtosis_x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__x"], ((((data["4__skewness_x"]) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (np.where(data["4__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["flux_d0_pb3"], data["2__kurtosis_x"] ))) )) +
                0.100000*np.tanh(((data["flux_d0_pb5"]) * (((((((data["3__skewness_y"]) + (data["2__kurtosis_y"]))) + (data["3__skewness_y"]))) + (((data["2__kurtosis_y"]) + (data["5__skewness_x"]))))))) +
                0.100000*np.tanh(((np.where(data["flux_dif3"]<0, data["detected_mjd_diff"], ((data["0__kurtosis_y"]) - (data["detected_mjd_diff"])) )) - (data["0__kurtosis_y"]))) +
                0.100000*np.tanh((-1.0*((np.where(((data["hostgal_photoz_err"]) * 2.0) > -1, (-1.0*((np.where(data["detected_flux_w_mean"]>0, data["flux_dif2"], data["5__fft_coefficient__coeff_1__attr__abs__x"] )))), data["flux_diff"] ))))) +
                0.100000*np.tanh(((data["detected_flux_ratio_sq_sum"]) + (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.where(data["flux_d1_pb1"] > -1, data["1__skewness_x"], (((data["1__fft_coefficient__coeff_1__attr__abs__x"]) > (data["flux_d1_pb0"]))*1.) )))

    def GP_class_52(self,data):
        return (-1.867467 +
                0.100000*np.tanh(((data["flux_by_flux_ratio_sq_sum"]) + (((((2.718282) + (data["flux_by_flux_ratio_sq_skew"]))) + (np.tanh((data["flux_by_flux_ratio_sq_sum"]))))))) +
                0.100000*np.tanh(((data["flux_min"]) + (((((data["3__skewness_x"]) + (data["flux_min"]))) + (data["flux_min"]))))) +
                0.100000*np.tanh((((((((((data["flux_min"]) + (((data["flux_min"]) + (data["flux_min"]))))) + (data["flux_d1_pb3"]))) + (data["flux_min"]))/2.0)) * 2.0)) +
                0.100000*np.tanh(((data["2__skewness_x"]) + (data["2__skewness_x"]))) +
                0.100000*np.tanh(((((data["flux_min"]) + (((data["flux_min"]) + (data["3__fft_coefficient__coeff_0__attr__abs__x"]))))) + (((data["2__skewness_x"]) + (data["2__skewness_x"]))))) +
                0.100000*np.tanh(np.where(data["flux_min"] > -1, data["flux_min"], ((((data["flux_min"]) + (data["detected_flux_by_flux_ratio_sq_skew"]))) + (data["flux_by_flux_ratio_sq_skew"])) )) +
                0.100000*np.tanh(((((data["flux_min"]) + (((((data["4__kurtosis_x"]) - (data["flux_diff"]))) * 2.0)))) + (data["flux_by_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((np.where(np.tanh((data["2__fft_coefficient__coeff_1__attr__abs__y"]))>0, data["flux_ratio_sq_skew"], data["2__skewness_x"] )) + (((data["mjd_size"]) - (data["detected_flux_err_mean"]))))) +
                0.100000*np.tanh(np.minimum(((data["3__kurtosis_x"])), ((np.minimum(((data["2__kurtosis_x"])), ((data["distmod"]))))))) +
                0.100000*np.tanh(((((np.minimum(((data["5__kurtosis_x"])), ((np.minimum(((data["5__kurtosis_x"])), ((data["5__kurtosis_x"]))))))) * 2.0)) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((data["flux_min"]) + (((data["5__kurtosis_x"]) + (data["distmod"]))))) +
                0.100000*np.tanh(((np.minimum(((((data["flux_d0_pb3"]) + (((np.where(data["4__kurtosis_x"]<0, data["flux_ratio_sq_skew"], data["flux_diff"] )) / 2.0))))), ((data["hostgal_photoz_err"])))) - (data["flux_diff"]))) +
                0.100000*np.tanh(np.minimum(((((data["flux_d0_pb2"]) / 2.0))), ((np.minimum(((data["flux_min"])), ((data["flux_err_skew"]))))))) +
                0.100000*np.tanh(((0.367879) * (data["0__kurtosis_x"]))) +
                0.100000*np.tanh(((((((data["distmod"]) + (data["flux_ratio_sq_skew"]))) - (((data["distmod"]) * (data["flux_err_std"]))))) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(np.minimum(((data["flux_d0_pb3"])), ((np.minimum(((data["2__skewness_x"])), (((((data["2__kurtosis_x"]) < (data["1__kurtosis_x"]))*1.)))))))) +
                0.100000*np.tanh(np.minimum(((((data["4__kurtosis_x"]) * 2.0))), ((data["distmod"])))) +
                0.100000*np.tanh((((((((-1.0*((data["flux_diff"])))) - (data["flux_diff"]))) - (((data["flux_diff"]) - (data["5__kurtosis_x"]))))) - (data["flux_diff"]))) +
                0.100000*np.tanh(np.minimum(((data["2__skewness_x"])), ((np.minimum(((data["4__kurtosis_x"])), ((data["flux_d0_pb3"]))))))) +
                0.100000*np.tanh(np.minimum(((data["0__kurtosis_x"])), ((data["5__kurtosis_x"])))) +
                0.100000*np.tanh(((((data["3__kurtosis_x"]) - (data["detected_flux_err_median"]))) + (((data["flux_std"]) - (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_std"]))))))) +
                0.100000*np.tanh(((data["5__kurtosis_x"]) - (data["flux_diff"]))) +
                0.100000*np.tanh(np.minimum(((data["flux_d0_pb3"])), ((((((data["4__skewness_x"]) - (((data["detected_flux_dif3"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])))))) +
                0.100000*np.tanh(((((data["hostgal_photoz_err"]) + (data["hostgal_photoz_err"]))) + (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((data["1__skewness_x"]) + (np.where(data["mjd_diff"]<0, data["3__fft_coefficient__coeff_1__attr__abs__x"], data["flux_d0_pb2"] )))) + (((data["flux_ratio_sq_skew"]) + (data["3__skewness_y"]))))) +
                0.100000*np.tanh((((((np.where(data["detected_flux_w_mean"]<0, data["flux_median"], data["flux_max"] )) < (data["flux_max"]))*1.)) - (data["flux_max"]))) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (data["distmod"]))) + (1.0))) +
                0.100000*np.tanh(((((data["detected_flux_diff"]) - (data["flux_err_max"]))) - (((data["detected_flux_std"]) * 2.0)))) +
                0.100000*np.tanh((((data["flux_diff"]) < (((data["flux_d0_pb2"]) - (data["flux_max"]))))*1.)) +
                0.100000*np.tanh(((((((((-1.0*((np.tanh((data["flux_d0_pb4"])))))) > (data["detected_flux_std"]))*1.)) * 2.0)) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((0.367879) + (((np.minimum(((((np.tanh((((data["flux_d0_pb2"]) + (data["detected_flux_min"]))))) + (data["flux_d0_pb3"])))), ((data["detected_flux_min"])))) * 2.0)))) +
                0.100000*np.tanh(((((((data["flux_median"]) + (data["flux_median"]))) + (data["flux_median"]))) + (((data["5__kurtosis_y"]) * (data["flux_median"]))))) +
                0.100000*np.tanh(((((data["1__kurtosis_y"]) + (((data["flux_median"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((((((((data["flux_d0_pb0"]) / 2.0)) / 2.0)) - (data["detected_flux_diff"]))) - (data["flux_d0_pb0"]))) +
                0.100000*np.tanh(((((((data["flux_median"]) + (((data["2__kurtosis_x"]) + (data["flux_ratio_sq_skew"]))))) + (np.tanh((data["flux_median"]))))) + (2.718282))) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) + (data["flux_d1_pb2"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, ((data["3__skewness_y"]) - (data["flux_d0_pb0"])), data["distmod"] )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb1"]<0, data["flux_d0_pb4"], ((data["2__kurtosis_x"]) * 2.0) )) +
                0.100000*np.tanh(((((((data["flux_median"]) * 2.0)) + (((((((data["flux_median"]) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((((((((data["hostgal_photoz_err"]) - (data["flux_d0_pb0"]))) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["flux_d0_pb0"]))) - (((data["flux_d0_pb0"]) - (data["flux_d0_pb0"]))))) +
                0.100000*np.tanh(((np.maximum(((data["distmod"])), ((data["flux_median"])))) + (((1.0) + (data["distmod"]))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["0__kurtosis_x"])), ((((data["flux_median"]) * 2.0)))))), ((data["4__kurtosis_x"])))) +
                0.100000*np.tanh(((np.tanh((((data["flux_mean"]) - (data["5__fft_coefficient__coeff_0__attr__abs__x"]))))) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_median"]) - (data["detected_flux_std"]))) + (data["flux_by_flux_ratio_sq_skew"]))) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((((((np.where(data["flux_d0_pb2"]>0, data["flux_d0_pb2"], data["flux_d0_pb2"] )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["5__kurtosis_x"]) - (data["detected_flux_diff"]))) +
                0.100000*np.tanh(((((np.where(data["4__fft_coefficient__coeff_1__attr__abs__x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__x"], data["5__kurtosis_x"] )) - (data["detected_flux_diff"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_d0_pb2"]>0, data["5__kurtosis_x"], data["5__kurtosis_x"] )) +
                0.100000*np.tanh(((data["flux_median"]) + ((((np.minimum(((data["5__kurtosis_y"])), ((np.tanh((data["flux_err_max"])))))) + (((data["flux_err_max"]) + (data["1__kurtosis_y"]))))/2.0)))) +
                0.100000*np.tanh(np.where(data["flux_d1_pb1"]>0, data["2__kurtosis_x"], np.where(data["flux_d1_pb1"]>0, data["ddf"], np.where(data["flux_d1_pb1"]>0, data["flux_d1_pb1"], (-1.0*((data["flux_d1_pb1"]))) ) ) )) +
                0.100000*np.tanh(((((data["hostgal_photoz_err"]) - (((data["detected_flux_std"]) - (np.where(data["5__fft_coefficient__coeff_0__attr__abs__y"] > -1, data["0__skewness_x"], data["flux_d1_pb4"] )))))) - (data["flux_d0_pb0"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_min"]>0, ((data["flux_median"]) * 2.0), data["mwebv"] )) +
                0.100000*np.tanh(((((((((((np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), ((data["2__kurtosis_x"])))) + (data["2__kurtosis_x"]))) * (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(np.where(data["flux_median"]<0, data["detected_flux_err_mean"], data["detected_flux_err_mean"] ) > -1, ((((data["mwebv"]) * 2.0)) * 2.0), data["3__fft_coefficient__coeff_0__attr__abs__x"] )) * 2.0)) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb2"]>0, data["4__fft_coefficient__coeff_1__attr__abs__x"], data["4__fft_coefficient__coeff_1__attr__abs__x"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_d1_pb2"]<0, data["flux_d0_pb0"], (((data["flux_d0_pb0"]) < (data["5__kurtosis_y"]))*1.) )) +
                0.100000*np.tanh(((((((np.where(data["flux_d0_pb0"]>0, data["4__kurtosis_x"], ((data["flux_median"]) - (data["2__kurtosis_x"])) )) - (data["flux_d0_pb0"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh((((((((data["flux_d0_pb2"]) > (data["flux_d1_pb1"]))*1.)) * 2.0)) + (((data["flux_d0_pb2"]) * 2.0)))) +
                0.100000*np.tanh(((np.where(data["flux_dif3"]<0, np.minimum(((((data["flux_ratio_sq_skew"]) + (data["flux_d1_pb2"])))), ((((data["flux_ratio_sq_skew"]) * 2.0)))), data["mwebv"] )) * 2.0)) +
                0.100000*np.tanh(((np.where(data["flux_d1_pb2"]>0, data["0__kurtosis_x"], data["4__skewness_y"] )) + (np.maximum(((data["4__skewness_y"])), ((data["flux_err_max"])))))) +
                0.100000*np.tanh(((((((data["mjd_diff"]) * (np.where(data["flux_d0_pb1"]<0, data["3__kurtosis_x"], np.where(data["flux_d0_pb1"]<0, data["mjd_diff"], data["5__fft_coefficient__coeff_0__attr__abs__x"] ) )))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["2__kurtosis_x"]) * (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(np.where((((data["detected_flux_max"]) < (data["flux_d0_pb2"]))*1.)>0, (((data["flux_d1_pb1"]) < (data["flux_d0_pb2"]))*1.), data["flux_err_std"] )) +
                0.100000*np.tanh(np.minimum(((((((data["flux_median"]) * (data["flux_by_flux_ratio_sq_skew"]))) * ((9.0))))), ((data["flux_d0_pb4"])))) +
                0.100000*np.tanh(np.where(data["flux_err_max"]>0, data["mjd_diff"], np.where(data["mjd_diff"]>0, data["0__kurtosis_x"], data["2__fft_coefficient__coeff_0__attr__abs__y"] ) )) +
                0.100000*np.tanh(((((np.where(data["flux_ratio_sq_skew"]<0, data["flux_err_max"], ((data["flux_d0_pb2"]) + (data["distmod"])) )) * 2.0)) + (data["flux_d0_pb4"]))) +
                0.100000*np.tanh(((((np.where(data["0__skewness_x"]<0, data["3__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["detected_mean"]>0, data["2__kurtosis_x"], ((data["0__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0) ) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh((((((((data["detected_flux_diff"]) > (data["flux_dif2"]))*1.)) - (((data["detected_flux_diff"]) - (data["flux_ratio_sq_skew"]))))) - (data["flux_max"]))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) * (((data["detected_flux_err_median"]) + (((((data["detected_flux_err_median"]) + (data["2__skewness_y"]))) + (data["2__skewness_y"]))))))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb1"]<0, ((data["flux_d0_pb0"]) * 2.0), data["2__kurtosis_x"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_min"]<0, data["detected_flux_min"], np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_err_max"], data["4__fft_coefficient__coeff_1__attr__abs__y"] ) )) +
                0.100000*np.tanh(((((((((((data["2__fft_coefficient__coeff_1__attr__abs__y"]) - (data["flux_d0_pb0"]))) * 2.0)) - (data["2__skewness_y"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((np.where(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)>0, data["4__kurtosis_x"], data["flux_d0_pb4"] )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["0__skewness_x"]>0, np.where(data["0__skewness_x"]>0, ((data["flux_d1_pb0"]) + (data["flux_err_max"])), ((data["flux_err_max"]) + (data["detected_flux_err_median"])) ), data["3__skewness_y"] )) +
                0.100000*np.tanh(((((data["2__kurtosis_x"]) * (((((((data["5__kurtosis_x"]) * (data["3__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)) * (((data["2__skewness_x"]) / 2.0)))))) * 2.0)) +
                0.100000*np.tanh((((((data["flux_median"]) > (data["2__skewness_y"]))*1.)) + (np.where(data["distmod"]>0, data["0__kurtosis_y"], ((data["distmod"]) + (data["flux_median"])) )))) +
                0.100000*np.tanh(((((np.minimum(((np.minimum(((np.minimum(((((data["3__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0))), ((data["4__kurtosis_x"]))))), ((data["2__kurtosis_x"]))))), ((data["flux_d1_pb1"])))) * 2.0)) * (data["flux_d1_pb1"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_std"]<0, data["flux_d0_pb3"], np.where(data["flux_median"]<0, data["0__kurtosis_x"], data["1__kurtosis_y"] ) )) +
                0.100000*np.tanh(((((((np.where(data["detected_flux_std"]>0, np.where(data["detected_flux_min"]<0, (-1.0*((data["3__fft_coefficient__coeff_0__attr__abs__y"]))), data["detected_flux_skew"] ), data["3__fft_coefficient__coeff_0__attr__abs__y"] )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]<0, ((data["4__skewness_y"]) - (((data["4__kurtosis_x"]) + (data["4__kurtosis_x"])))), data["4__kurtosis_x"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_mean"]>0, data["4__kurtosis_x"], np.where(data["distmod"]>0, np.where(data["hostgal_photoz_err"]<0, data["detected_mean"], data["hostgal_photoz_err"] ), data["0__skewness_x"] ) )) +
                0.100000*np.tanh(np.where(data["flux_std"]>0, np.where(data["hostgal_photoz_err"]>0, data["distmod"], (-1.0*((data["1__fft_coefficient__coeff_0__attr__abs__x"]))) ), data["hostgal_photoz_err"] )) +
                0.100000*np.tanh(np.where(np.where(data["flux_err_median"]<0, data["detected_mean"], data["2__fft_coefficient__coeff_0__attr__abs__y"] )<0, data["flux_d1_pb0"], data["2__kurtosis_x"] )) +
                0.100000*np.tanh(np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"]>0, np.where(np.tanh((data["4__kurtosis_y"]))>0, data["2__fft_coefficient__coeff_0__attr__abs__y"], data["1__kurtosis_y"] ), ((data["flux_d1_pb4"]) + (data["detected_mean"])) )) +
                0.100000*np.tanh(np.where(data["flux_median"]<0, np.where(data["1__kurtosis_y"]<0, data["2__fft_coefficient__coeff_1__attr__abs__x"], ((data["1__skewness_x"]) + (data["flux_median"])) ), data["1__kurtosis_y"] )) +
                0.100000*np.tanh(((data["2__kurtosis_y"]) * (np.where(np.tanh((data["2__skewness_y"])) > -1, data["flux_diff"], data["2__skewness_y"] )))) +
                0.100000*np.tanh(((data["flux_d1_pb4"]) + (np.where(data["detected_flux_err_mean"] > -1, np.where(data["0__kurtosis_x"] > -1, data["4__kurtosis_y"], data["flux_d1_pb4"] ), data["3__fft_coefficient__coeff_0__attr__abs__x"] )))) +
                0.100000*np.tanh(((((data["2__fft_coefficient__coeff_0__attr__abs__x"]) - (data["3__skewness_x"]))) * (((data["2__skewness_y"]) + (np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"] > -1, ((data["2__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0), data["3__skewness_x"] )))))) +
                0.100000*np.tanh(((data["distmod"]) + (((data["2__kurtosis_x"]) * (((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["distmod"]))) * 2.0)))) * 2.0)))))) +
                0.100000*np.tanh(((data["flux_err_max"]) + (((((np.where(data["detected_mean"]>0, data["2__kurtosis_x"], ((data["flux_err_median"]) + (data["2__kurtosis_x"])) )) * 2.0)) + (data["detected_mean"]))))) +
                0.100000*np.tanh(((np.where(data["detected_flux_mean"]<0, data["3__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["0__fft_coefficient__coeff_0__attr__abs__x"], data["flux_ratio_sq_skew"] ), data["flux_ratio_sq_skew"] ) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_min"] > -1, ((data["detected_flux_skew"]) * (np.where(data["detected_flux_min"]>0, data["detected_flux_ratio_sq_skew"], data["detected_flux_err_median"] ))), ((data["detected_flux_err_median"]) * (data["0__fft_coefficient__coeff_1__attr__abs__x"])) )) +
                0.100000*np.tanh(np.where(data["detected_flux_err_skew"]<0, data["flux_median"], np.where(data["flux_median"]<0, data["1__skewness_x"], data["5__kurtosis_y"] ) )) +
                0.100000*np.tanh(np.where(data["0__kurtosis_x"]<0, data["3__skewness_y"], np.where(data["3__skewness_y"]<0, data["0__skewness_x"], np.where(data["0__kurtosis_x"]>0, data["0__kurtosis_x"], data["2__kurtosis_x"] ) ) )) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, np.where(data["detected_flux_err_mean"] > -1, np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), (((4.0)))), (4.0) ), (4.0) )) +
                0.100000*np.tanh(((((np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"]<0, data["distmod"], data["detected_flux_err_max"] )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_err_skew"] > -1, np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_err_max"], data["detected_flux_err_skew"] ), ((data["detected_flux_err_skew"]) * 2.0) )) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_1__attr__abs__y"]<0, ((data["3__kurtosis_x"]) * (((((data["flux_d0_pb0"]) * 2.0)) * 2.0))), data["3__kurtosis_x"] )) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb0"]>0, data["3__kurtosis_x"], ((((data["4__skewness_y"]) - (data["3__kurtosis_x"]))) - (data["3__kurtosis_x"])) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["mjd_size"]>0, ((np.where(data["mjd_size"]>0, data["1__kurtosis_y"], ((data["detected_flux_err_min"]) + (data["flux_err_mean"])) )) + (data["flux_err_mean"])), data["3__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(((np.where(data["0__skewness_x"]>0, ((np.where(data["0__skewness_x"]>0, data["hostgal_photoz_err"], data["flux_d1_pb2"] )) - (data["flux_d1_pb2"])), data["flux_d1_pb2"] )) * 2.0)) +
                0.100000*np.tanh(((((((np.where(data["4__fft_coefficient__coeff_1__attr__abs__x"]<0, ((data["4__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0), np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]<0, data["flux_d0_pb2"], data["5__kurtosis_x"] ) )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_mean"]<0, ((((data["3__kurtosis_x"]) - (data["4__kurtosis_x"]))) - (data["2__skewness_y"])), data["2__skewness_y"] )) +
                0.100000*np.tanh(((((np.where(data["detected_flux_err_mean"] > -1, (((((data["flux_err_max"]) > (data["2__fft_coefficient__coeff_0__attr__abs__x"]))*1.)) - (data["mjd_size"])), data["2__fft_coefficient__coeff_0__attr__abs__x"] )) * 2.0)) * 2.0)) +
                0.100000*np.tanh((-1.0*((((np.where(data["0__skewness_x"] > -1, np.where(data["0__skewness_x"] > -1, data["0__skewness_x"], data["2__skewness_y"] ), data["4__kurtosis_y"] )) * (((data["2__skewness_y"]) * 2.0))))))) +
                0.100000*np.tanh(((((data["2__kurtosis_x"]) * (((((data["flux_d1_pb2"]) + (data["distmod"]))) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(np.where(data["1__kurtosis_x"]<0, np.where(data["detected_flux_max"]<0, data["detected_flux_max"], data["detected_flux_max"] ), ((((((data["detected_flux_max"]) < (data["detected_mjd_size"]))*1.)) < (data["flux_d0_pb1"]))*1.) )) +
                0.100000*np.tanh(((np.where(data["flux_err_max"]<0, data["5__fft_coefficient__coeff_1__attr__abs__y"], np.where(data["distmod"]<0, data["detected_flux_err_skew"], (((-1.0*((data["5__fft_coefficient__coeff_1__attr__abs__y"])))) * 2.0) ) )) * 2.0)) +
                0.100000*np.tanh((((11.19984531402587891)) * (((data["distmod"]) + ((((data["flux_w_mean"]) < ((((data["5__kurtosis_x"]) > ((((data["flux_w_mean"]) > (data["5__kurtosis_x"]))*1.)))*1.)))*1.)))))) +
                0.100000*np.tanh(((np.where(data["0__kurtosis_x"]<0, data["flux_err_max"], np.where(data["0__kurtosis_x"]<0, data["detected_flux_err_std"], np.where(data["flux_err_max"]<0, data["1__fft_coefficient__coeff_0__attr__abs__x"], data["detected_flux_err_std"] ) ) )) * 2.0)) +
                0.100000*np.tanh(((((np.where(data["detected_flux_err_median"]>0, data["5__fft_coefficient__coeff_1__attr__abs__x"], ((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)) * (data["hostgal_photoz"])) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((((((np.where(data["flux_ratio_sq_skew"]<0, data["3__fft_coefficient__coeff_1__attr__abs__y"], np.where(data["4__fft_coefficient__coeff_1__attr__abs__x"]<0, data["distmod"], data["2__fft_coefficient__coeff_0__attr__abs__y"] ) )) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_mean"]>0, np.where(data["detected_mean"]<0, data["2__skewness_y"], data["2__skewness_y"] ), np.where(data["flux_ratio_sq_sum"]>0, (-1.0*((data["2__skewness_y"]))), data["detected_mean"] ) )) +
                0.100000*np.tanh(np.where(data["flux_ratio_sq_skew"]<0, data["hostgal_photoz_err"], np.maximum(((data["flux_d1_pb0"])), ((np.where(data["flux_ratio_sq_skew"]<0, data["hostgal_photoz_err"], np.maximum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((data["flux_d1_pb0"]))) )))) )) +
                0.100000*np.tanh(np.where(data["0__kurtosis_x"]<0, data["3__fft_coefficient__coeff_1__attr__abs__x"], np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]>0, data["3__fft_coefficient__coeff_1__attr__abs__x"], np.where(data["3__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["flux_d1_pb0"], data["5__fft_coefficient__coeff_0__attr__abs__y"] ) ) )) +
                0.100000*np.tanh(np.where(data["flux_dif2"]<0, data["1__skewness_y"], ((((((data["flux_ratio_sq_skew"]) - (data["1__skewness_y"]))) - (data["detected_flux_by_flux_ratio_sq_skew"]))) - (data["flux_dif2"])) )) +
                0.100000*np.tanh(((((np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"]<0, ((data["flux_ratio_sq_skew"]) * 2.0), ((data["4__skewness_y"]) - (data["5__fft_coefficient__coeff_0__attr__abs__y"])) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["2__fft_coefficient__coeff_0__attr__abs__y"])), ((data["flux_d1_pb2"])))) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, ((np.where(data["3__fft_coefficient__coeff_1__attr__abs__x"]>0, data["hostgal_photoz_err"], np.where(data["flux_d1_pb4"] > -1, data["flux_d1_pb4"], data["flux_d1_pb4"] ) )) * 2.0), 3.141593 )) +
                0.100000*np.tanh(((np.where(data["flux_err_skew"]>0, np.where(data["1__fft_coefficient__coeff_0__attr__abs__x"]>0, data["flux_err_skew"], data["1__fft_coefficient__coeff_0__attr__abs__x"] ), np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]>0, data["5__fft_coefficient__coeff_1__attr__abs__x"], data["5__kurtosis_x"] ) )) * 2.0)))

    def GP_class_53(self,data):
        return (-2.781493 +
                0.100000*np.tanh(((data["flux_err_std"]) + (np.minimum(((data["2__skewness_y"])), ((data["5__fft_coefficient__coeff_1__attr__abs__y"])))))) +
                0.100000*np.tanh(np.where(data["3__fft_coefficient__coeff_0__attr__abs__x"]<0, data["detected_mean"], data["detected_flux_err_median"] )) +
                0.100000*np.tanh((((((((((((data["flux_err_mean"]) + (((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (data["2__fft_coefficient__coeff_1__attr__abs__y"]))))/2.0)) + (data["5__skewness_y"]))/2.0)) + (-1.0))/2.0)) + (data["flux_err_std"]))) +
                0.100000*np.tanh(np.minimum(((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (data["5__fft_coefficient__coeff_0__attr__abs__x"])))), ((data["3__skewness_y"])))) +
                0.100000*np.tanh((((data["flux_err_std"]) + (np.maximum(((data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((np.minimum(((((data["detected_mean"]) + (data["2__fft_coefficient__coeff_0__attr__abs__y"])))), ((data["5__fft_coefficient__coeff_1__attr__abs__y"]))))))))/2.0)) +
                0.100000*np.tanh(((np.minimum(((data["5__skewness_y"])), ((((data["flux_err_mean"]) + (data["detected_mean"])))))) + (np.where(data["detected_flux_diff"] > -1, data["flux_err_mean"], -2.0 )))) +
                0.100000*np.tanh(np.minimum(((data["detected_mean"])), ((((np.where(data["2__skewness_y"] > -1, data["3__skewness_y"], data["flux_err_max"] )) + (((data["1__kurtosis_x"]) / 2.0))))))) +
                0.100000*np.tanh(((((0.0) + (data["5__skewness_y"]))) + (np.where(-1.0<0, data["flux_err_mean"], data["flux_d1_pb1"] )))) +
                0.100000*np.tanh(((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + ((((np.minimum(((data["detected_flux_std"])), ((data["5__fft_coefficient__coeff_1__attr__abs__y"])))) + ((((data["5__skewness_y"]) + (data["flux_max"]))/2.0)))/2.0)))) +
                0.100000*np.tanh(np.minimum(((data["detected_mean"])), (((((data["detected_mean"]) + (data["flux_err_std"]))/2.0))))) +
                0.100000*np.tanh(((((data["flux_err_std"]) + (data["flux_err_std"]))) + (data["3__skewness_y"]))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (-3.0))) + (data["3__skewness_y"]))) +
                0.100000*np.tanh(np.where(data["2__fft_coefficient__coeff_1__attr__abs__y"] > -1, np.where(data["flux_err_std"] > -1, data["flux_err_std"], data["5__fft_coefficient__coeff_1__attr__abs__y"] ), data["2__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(((np.where(data["flux_err_std"]>0, data["flux_err_std"], data["3__skewness_y"] )) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.where(data["detected_mean"] > -1, data["flux_err_std"], data["3__skewness_y"] )) +
                0.100000*np.tanh(((((data["5__skewness_y"]) + (((data["5__skewness_y"]) * 2.0)))) + (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((np.minimum(((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) - (data["0__fft_coefficient__coeff_0__attr__abs__x"])))), ((data["0__fft_coefficient__coeff_0__attr__abs__y"])))) + (((data["5__skewness_y"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(((((((data["flux_err_std"]) + (data["flux_err_std"]))) + (((data["3__skewness_y"]) + (((data["1__kurtosis_x"]) / 2.0)))))) + (-2.0))) +
                0.100000*np.tanh(np.minimum(((np.minimum((((((data["3__skewness_y"]) + (data["flux_err_std"]))/2.0))), ((data["detected_mean"]))))), ((((data["4__skewness_y"]) + (data["4__skewness_y"])))))) +
                0.100000*np.tanh(np.minimum(((data["3__skewness_y"])), ((((data["2__fft_coefficient__coeff_1__attr__abs__y"]) + (data["3__kurtosis_y"])))))) +
                0.100000*np.tanh(((((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((-3.0)))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh((((((data["3__skewness_y"]) + (((((-3.0) + (data["3__skewness_y"]))) + (data["flux_err_std"]))))/2.0)) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["4__skewness_y"])), ((data["4__skewness_y"]))))), ((((data["3__kurtosis_y"]) + (data["flux_err_mean"])))))) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]>0, -3.0, ((np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]>0, -3.0, data["flux_err_std"] )) + (((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (3.141593)))) )) +
                0.100000*np.tanh(np.minimum(((data["3__fft_coefficient__coeff_1__attr__abs__y"])), ((np.minimum(((data["2__fft_coefficient__coeff_0__attr__abs__y"])), ((((np.minimum(((((data["3__kurtosis_y"]) + (data["3__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["3__kurtosis_y"])))) + (data["detected_flux_err_median"]))))))))) +
                0.100000*np.tanh(((((data["3__kurtosis_y"]) + (-2.0))) + (((np.minimum(((data["flux_err_std"])), ((data["flux_err_std"])))) + (-2.0))))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((np.minimum(((-3.0)), ((-3.0))))))))) +
                0.100000*np.tanh(((data["flux_err_std"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((np.minimum(((((data["4__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0))), ((np.maximum(((data["detected_mean"])), ((((data["4__fft_coefficient__coeff_1__attr__abs__x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))))))) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (np.where(data["detected_mean"] > -1, -3.0, ((data["flux_std"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])) )))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"] > -1, 3.141593, np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]>0, data["0__fft_coefficient__coeff_1__attr__abs__x"], 3.141593 ) )))) +
                0.100000*np.tanh(np.minimum(((data["flux_err_std"])), ((((np.minimum(((np.minimum(((data["5__fft_coefficient__coeff_0__attr__abs__y"])), ((np.minimum(((data["3__skewness_y"])), ((data["flux_err_std"])))))))), ((data["3__skewness_y"])))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])))))) +
                0.100000*np.tanh(((np.where(data["detected_mean"] > -1, data["detected_mean"], np.where(data["detected_flux_err_max"]>0, data["flux_err_std"], data["detected_flux_err_mean"] ) )) - (2.0))) +
                0.100000*np.tanh(((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) - (((data["flux_max"]) * (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((((((-2.0) + (((-2.0) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))))) + (data["flux_err_std"]))) + (data["5__fft_coefficient__coeff_0__attr__abs__x"])))), ((data["detected_mean"])))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["3__kurtosis_y"])), ((data["flux_err_std"]))))), ((data["5__fft_coefficient__coeff_0__attr__abs__x"])))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__x"], 3.141593 )))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - ((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_err_std"]))/2.0)) - (data["flux_err_std"]))))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["4__fft_coefficient__coeff_1__attr__abs__x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) - (2.0))) +
                0.100000*np.tanh(((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_err_std"]))) + (data["flux_d1_pb0"]))) +
                0.100000*np.tanh(((np.minimum(((data["2__kurtosis_y"])), (((((((data["3__kurtosis_y"]) + (data["2__kurtosis_y"]))/2.0)) + (data["flux_d1_pb1"])))))) + (data["flux_diff"]))) +
                0.100000*np.tanh(np.minimum(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (3.0)))), ((data["detected_mjd_diff"])))) +
                0.100000*np.tanh(np.minimum(((data["2__kurtosis_y"])), ((np.minimum(((data["2__kurtosis_y"])), ((((np.minimum(((data["flux_max"])), ((((data["4__skewness_y"]) * 2.0))))) * 2.0)))))))) +
                0.100000*np.tanh(((np.minimum(((((data["flux_err_std"]) - (data["2__fft_coefficient__coeff_0__attr__abs__y"])))), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(data["3__skewness_x"] > -1, data["flux_d0_pb1"], data["flux_d1_pb1"] )) +
                0.100000*np.tanh(np.minimum(((data["flux_d0_pb1"])), ((((data["flux_max"]) - (data["3__kurtosis_y"])))))) +
                0.100000*np.tanh(((np.where(data["flux_err_max"]>0, ((data["flux_max"]) - (data["detected_flux_ratio_sq_skew"])), ((data["flux_max"]) + (data["0__fft_coefficient__coeff_0__attr__abs__x"])) )) - ((-1.0*((data["flux_err_median"])))))) +
                0.100000*np.tanh(((-2.0) + (np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]>0, np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]>0, -2.0, data["0__fft_coefficient__coeff_0__attr__abs__y"] ), data["detected_mean"] )))) +
                0.100000*np.tanh(((((data["2__skewness_x"]) - (data["detected_flux_ratio_sq_skew"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(np.minimum(((((((data["flux_err_mean"]) - (data["3__kurtosis_y"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__y"])))), ((data["flux_max"])))) +
                0.100000*np.tanh(np.where(data["flux_max"]<0, data["3__kurtosis_y"], np.minimum(((np.minimum(((data["3__kurtosis_y"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"]))))), ((data["3__kurtosis_y"]))) )) +
                0.100000*np.tanh((((data["4__skewness_y"]) + (np.where(data["3__skewness_x"]>0, np.where(-1.0>0, data["2__fft_coefficient__coeff_0__attr__abs__x"], data["2__fft_coefficient__coeff_1__attr__abs__x"] ), data["flux_d0_pb3"] )))/2.0)) +
                0.100000*np.tanh(np.minimum(((data["flux_max"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh((((((-1.0*((data["detected_flux_ratio_sq_skew"])))) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) +
                0.100000*np.tanh((-1.0*((np.where(data["0__skewness_x"] > -1, np.where(data["0__skewness_x"] > -1, data["0__skewness_x"], data["0__skewness_x"] ), np.where(data["0__skewness_x"]>0, data["0__skewness_x"], data["0__skewness_x"] ) ))))) +
                0.100000*np.tanh(np.where(((data["1__fft_coefficient__coeff_0__attr__abs__y"]) * (data["0__skewness_x"]))<0, ((data["2__fft_coefficient__coeff_1__attr__abs__x"]) / 2.0), data["0__fft_coefficient__coeff_0__attr__abs__x"] )) +
                0.100000*np.tanh(np.minimum(((((((np.minimum(((data["flux_d0_pb1"])), ((data["flux_err_mean"])))) + (-2.0))) + (data["detected_mean"])))), ((data["hostgal_photoz_err"])))) +
                0.100000*np.tanh((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) + (np.where(((data["1__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)<0, data["3__kurtosis_y"], ((data["1__kurtosis_x"]) * 2.0) )))/2.0)) +
                0.100000*np.tanh(np.minimum(((np.where(data["flux_d0_pb1"]<0, data["flux_d0_pb1"], np.where(data["flux_d0_pb1"]>0, data["detected_mean"], data["flux_d0_pb1"] ) ))), ((np.minimum(((data["detected_mean"])), ((data["2__fft_coefficient__coeff_0__attr__abs__y"]))))))) +
                0.100000*np.tanh(((np.minimum(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["flux_std"]) - (data["5__fft_coefficient__coeff_0__attr__abs__x"])))))), ((((-3.0) - (data["0__fft_coefficient__coeff_0__attr__abs__x"])))))) + (data["flux_std"]))) +
                0.100000*np.tanh(np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__x"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))) +
                0.100000*np.tanh((((((data["flux_err_std"]) + (np.tanh((data["flux_d1_pb0"]))))/2.0)) - (((data["flux_d1_pb2"]) * (data["flux_d0_pb0"]))))) +
                0.100000*np.tanh(((((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) - (((data["0__kurtosis_x"]) - (data["2__fft_coefficient__coeff_0__attr__abs__y"]))))) - (data["flux_err_skew"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) * (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])))), ((((data["flux_d1_pb1"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))))))), ((data["detected_mean"])))) +
                0.100000*np.tanh(((((((((((((data["flux_err_median"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) * 2.0)) * 2.0)) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.maximum((((((np.where(data["3__kurtosis_y"]>0, data["3__skewness_y"], data["3__kurtosis_y"] )) + ((((data["3__kurtosis_y"]) + (data["detected_flux_diff"]))/2.0)))/2.0))), ((np.tanh((data["detected_mean"])))))) +
                0.100000*np.tanh(((np.where(data["detected_flux_min"]<0, data["3__kurtosis_y"], np.minimum(((3.141593)), ((np.minimum(((-2.0)), (((((data["2__kurtosis_x"]) + (data["1__skewness_y"]))/2.0))))))) )) / 2.0)) +
                0.100000*np.tanh(np.minimum(((data["flux_d0_pb1"])), (((((((-1.0*((np.tanh((data["detected_flux_by_flux_ratio_sq_sum"])))))) / 2.0)) / 2.0))))) +
                0.100000*np.tanh(np.where(data["3__kurtosis_y"] > -1, ((np.where(data["3__kurtosis_y"]>0, data["flux_d0_pb1"], np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"]<0, data["0__skewness_y"], data["flux_d0_pb1"] ) )) * 2.0), data["flux_err_std"] )) +
                0.100000*np.tanh((((np.tanh(((((data["ddf"]) + (data["2__skewness_x"]))/2.0)))) + (data["detected_flux_err_max"]))/2.0)) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["3__skewness_y"])), ((np.minimum(((data["flux_err_max"])), ((data["3__kurtosis_y"])))))))), ((data["hostgal_photoz_err"])))) +
                0.100000*np.tanh((-1.0*((np.where(data["0__kurtosis_y"] > -1, ((data["0__skewness_x"]) - (((data["detected_flux_max"]) - (data["0__skewness_x"])))), data["2__fft_coefficient__coeff_1__attr__abs__x"] ))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["flux_max"])), ((data["3__kurtosis_y"]))))), ((((np.maximum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), ((data["detected_mean"])))) / 2.0))))) +
                0.100000*np.tanh((((((data["3__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["detected_mjd_diff"]) / 2.0)))/2.0)) * (np.minimum(((data["flux_err_max"])), ((data["3__fft_coefficient__coeff_1__attr__abs__x"])))))) +
                0.100000*np.tanh(np.minimum(((((-3.0) + (data["5__fft_coefficient__coeff_0__attr__abs__x"])))), ((((np.where(-3.0>0, data["2__skewness_x"], data["mjd_diff"] )) + (data["3__fft_coefficient__coeff_1__attr__abs__y"])))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((data["flux_ratio_sq_sum"]))))), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh(np.minimum(((((data["1__kurtosis_y"]) + (((data["mjd_diff"]) - (((data["2__skewness_y"]) / 2.0))))))), ((data["detected_flux_max"])))) +
                0.100000*np.tanh(np.minimum(((((data["2__kurtosis_x"]) + (data["3__kurtosis_y"])))), ((data["3__kurtosis_y"])))) +
                0.100000*np.tanh(np.minimum(((((((((data["4__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0)) * (data["3__fft_coefficient__coeff_0__attr__abs__y"]))) * (((data["4__skewness_y"]) * 2.0))))), ((data["5__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh((((((3.141593) < (np.where(data["ddf"]<0, data["3__fft_coefficient__coeff_1__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] )))*1.)) / 2.0)) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["2__fft_coefficient__coeff_0__attr__abs__y"])), ((data["2__skewness_y"]))))), ((((((data["flux_ratio_sq_sum"]) / 2.0)) - (data["flux_dif2"])))))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_max"])))), ((((data["4__skewness_y"]) * 2.0))))) +
                0.100000*np.tanh((-1.0*((np.where(data["detected_flux_w_mean"] > -1, data["0__skewness_x"], data["0__skewness_x"] ))))) +
                0.100000*np.tanh(np.minimum(((data["3__skewness_y"])), ((((np.tanh((np.maximum(((np.minimum(((data["2__fft_coefficient__coeff_0__attr__abs__y"])), ((data["2__fft_coefficient__coeff_1__attr__abs__x"]))))), ((data["mjd_size"])))))) - (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) / 2.0))))))) +
                0.100000*np.tanh(np.minimum(((((np.minimum(((((data["4__skewness_y"]) / 2.0))), ((((data["flux_d1_pb1"]) - (data["2__skewness_y"])))))) * 2.0))), ((np.minimum(((data["detected_flux_ratio_sq_sum"])), ((data["3__fft_coefficient__coeff_0__attr__abs__x"]))))))) +
                0.100000*np.tanh(((3.0) * ((((data["2__kurtosis_y"]) + (data["detected_mean"]))/2.0)))) +
                0.100000*np.tanh(np.where(np.minimum(((data["4__fft_coefficient__coeff_0__attr__abs__x"])), ((np.minimum(((data["2__kurtosis_y"])), ((np.tanh((data["flux_median"]))))))))>0, ((data["flux_max"]) / 2.0), data["2__skewness_y"] )) +
                0.100000*np.tanh(np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__y"])), ((data["flux_err_std"])))) +
                0.100000*np.tanh((-1.0*((((((data["2__skewness_x"]) - ((((data["2__kurtosis_y"]) < (data["0__kurtosis_y"]))*1.)))) / 2.0))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((data["5__fft_coefficient__coeff_0__attr__abs__x"])), (((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["2__fft_coefficient__coeff_1__attr__abs__y"]))/2.0)))))), ((data["2__skewness_y"]))))), ((data["2__skewness_x"])))) +
                0.100000*np.tanh(((np.where((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) < (data["detected_flux_dif3"]))*1.) > -1, data["2__fft_coefficient__coeff_1__attr__abs__x"], data["1__fft_coefficient__coeff_1__attr__abs__y"] )) - (data["4__skewness_x"]))) +
                0.100000*np.tanh(np.where(data["flux_ratio_sq_skew"]<0, data["detected_flux_err_max"], data["2__kurtosis_y"] )) +
                0.100000*np.tanh(np.minimum((((-1.0*((data["detected_flux_std"]))))), ((((data["hostgal_photoz"]) + ((-1.0*((((3.141593) * (data["detected_flux_err_median"]))))))))))) +
                0.100000*np.tanh(np.where(((((3.67651557922363281)) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))/2.0) > -1, (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_max"]))/2.0), data["flux_d1_pb1"] )) +
                0.100000*np.tanh(np.where(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"]<0, data["2__kurtosis_y"], np.where(data["1__skewness_y"]>0, data["flux_d1_pb2"], data["flux_d1_pb2"] ) )>0, 2.718282, data["flux_d1_pb2"] )) +
                0.100000*np.tanh(np.where(data["flux_mean"] > -1, data["flux_d1_pb1"], data["2__kurtosis_y"] )) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_0__attr__abs__x"]) * (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(np.minimum(((((data["2__skewness_x"]) + (((data["3__kurtosis_y"]) * 2.0))))), ((data["3__kurtosis_y"])))) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (((2.718282) - (data["detected_flux_by_flux_ratio_sq_sum"]))))))) +
                0.100000*np.tanh(((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + ((((data["detected_mjd_diff"]) + (data["detected_flux_dif2"]))/2.0)))) + (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.where(data["flux_d1_pb1"] > -1, data["1__kurtosis_x"], data["flux_median"] )) - (((data["0__skewness_y"]) * (data["2__fft_coefficient__coeff_0__attr__abs__x"]))))) +
                0.100000*np.tanh(((data["flux_max"]) + ((((np.tanh((((data["flux_err_std"]) + (0.0))))) + (data["flux_median"]))/2.0)))) +
                0.100000*np.tanh(((data["5__skewness_y"]) + (data["1__skewness_y"]))) +
                0.100000*np.tanh((((data["detected_flux_mean"]) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))/2.0)) +
                0.100000*np.tanh(np.tanh((((data["detected_flux_err_max"]) - ((-1.0*((np.minimum(((((data["detected_flux_err_max"]) / 2.0))), ((data["2__fft_coefficient__coeff_1__attr__abs__y"]))))))))))) +
                0.100000*np.tanh(((np.minimum(((((-3.0) - ((((0.0)) + (2.718282)))))), ((data["0__fft_coefficient__coeff_1__attr__abs__x"])))) - (data["detected_flux_diff"]))) +
                0.100000*np.tanh((((((data["detected_flux_std"]) * (((((np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__x"])), ((data["3__skewness_x"])))) + (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))))) + (0.367879))/2.0)) +
                0.100000*np.tanh(np.where(data["3__kurtosis_y"]<0, data["3__kurtosis_y"], (((((np.minimum(((data["3__kurtosis_y"])), ((data["3__kurtosis_y"])))) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) - (((data["flux_err_max"]) * 2.0))) )) +
                0.100000*np.tanh(np.maximum(((data["detected_flux_err_max"])), ((data["detected_mean"])))) +
                0.100000*np.tanh(((np.tanh((data["flux_ratio_sq_sum"]))) - (np.where(data["0__kurtosis_y"]>0, (((data["flux_ratio_sq_skew"]) + (data["4__skewness_y"]))/2.0), data["flux_err_skew"] )))) +
                0.100000*np.tanh(((data["2__fft_coefficient__coeff_0__attr__abs__y"]) - (((((data["flux_mean"]) / 2.0)) - (data["flux_err_std"]))))) +
                0.100000*np.tanh(((data["flux_median"]) - ((((-1.0*((data["5__skewness_x"])))) - (((data["flux_dif2"]) * (data["flux_by_flux_ratio_sq_skew"]))))))) +
                0.100000*np.tanh(np.where(np.minimum(((data["3__skewness_y"])), ((data["3__fft_coefficient__coeff_1__attr__abs__y"])))<0, np.tanh((((np.tanh((data["3__fft_coefficient__coeff_0__attr__abs__x"]))) / 2.0))), data["1__kurtosis_y"] )) +
                0.100000*np.tanh(np.minimum(((np.where(np.minimum(((((data["2__kurtosis_x"]) / 2.0))), ((data["detected_flux_by_flux_ratio_sq_sum"])))<0, data["flux_d0_pb2"], data["3__kurtosis_y"] ))), ((-3.0)))) +
                0.100000*np.tanh(np.minimum(((np.where((((data["flux_d1_pb2"]) + (data["2__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)<0, data["5__skewness_y"], data["2__fft_coefficient__coeff_0__attr__abs__x"] ))), ((data["3__skewness_y"])))) +
                0.100000*np.tanh(np.where(data["detected_flux_diff"] > -1, data["2__kurtosis_y"], np.where(data["1__fft_coefficient__coeff_1__attr__abs__y"] > -1, data["3__kurtosis_y"], data["2__skewness_y"] ) )) +
                0.100000*np.tanh(np.where(data["mwebv"] > -1, data["2__fft_coefficient__coeff_0__attr__abs__x"], data["2__fft_coefficient__coeff_0__attr__abs__x"] )) +
                0.100000*np.tanh(np.minimum(((data["flux_err_max"])), ((data["detected_flux_err_median"])))) +
                0.100000*np.tanh((((((((((np.tanh((data["flux_diff"]))) / 2.0)) * 2.0)) + (data["1__kurtosis_x"]))/2.0)) / 2.0)) +
                0.100000*np.tanh(np.where(data["0__kurtosis_x"] > -1, (-1.0*((np.minimum(((data["flux_d0_pb1"])), ((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) / 2.0))))))), -2.0 )))

    def GP_class_62(self,data):
        return (-1.361137 +
                0.100000*np.tanh(np.where(data["distmod"] > -1, ((data["flux_d0_pb5"]) + (((((data["detected_flux_median"]) + (((data["5__skewness_x"]) * 2.0)))) + (data["distmod"])))), -3.0 )) +
                0.100000*np.tanh(((data["4__kurtosis_x"]) + (((((np.maximum(((data["detected_flux_min"])), ((data["detected_flux_min"])))) + (((data["4__kurtosis_x"]) + (data["detected_flux_min"]))))) + (data["3__kurtosis_x"]))))) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["2__kurtosis_x"])), ((data["flux_min"]))))), ((data["5__kurtosis_x"])))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((data["flux_d1_pb4"]) + (data["5__skewness_x"])))), ((data["5__skewness_x"])))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((data["hostgal_photoz_err"])), ((data["flux_d0_pb5"]))))), ((data["flux_d0_pb5"]))))), ((np.minimum(((data["flux_min"])), ((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((data["5__skewness_x"])))))))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((data["flux_min"])), ((np.minimum(((data["flux_d0_pb5"])), ((((data["flux_d0_pb5"]) + (data["flux_d0_pb5"])))))))))), ((data["detected_flux_min"]))))), ((data["flux_min"])))) +
                0.100000*np.tanh(np.minimum(((data["flux_min"])), ((1.0)))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["flux_d0_pb1"])), ((np.tanh((data["flux_d0_pb5"]))))))), ((np.minimum(((data["1__skewness_x"])), ((data["1__kurtosis_x"]))))))) +
                0.100000*np.tanh(np.minimum(((data["flux_min"])), ((data["flux_min"])))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((-3.0)), ((-3.0))))), ((data["0__kurtosis_x"])))) +
                0.100000*np.tanh(((((((data["distmod"]) - (np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__x"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))))) - (data["detected_flux_err_std"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), ((((np.minimum(((data["detected_flux_mean"])), ((data["hostgal_photoz_err"])))) - (((data["hostgal_photoz_err"]) - (-1.0))))))))), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh(np.minimum((((((((data["detected_flux_min"]) + (np.minimum(((data["5__skewness_x"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))))/2.0)) + (data["flux_d0_pb4"])))), ((data["4__skewness_x"])))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["0__skewness_y"])), ((np.minimum(((data["detected_flux_min"])), ((data["detected_mjd_diff"])))))))), ((data["flux_dif2"])))) +
                0.100000*np.tanh(((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__y"])), ((((data["flux_dif2"]) * 2.0))))) + (data["4__kurtosis_x"]))) +
                0.100000*np.tanh(np.minimum(((((data["3__kurtosis_x"]) * 2.0))), (((((((data["5__kurtosis_x"]) < (data["4__skewness_x"]))*1.)) / 2.0))))) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_min"])), ((((data["detected_flux_min"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"])))))) +
                0.100000*np.tanh(((((data["detected_flux_min"]) - (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["hostgal_photoz_err"]))))) - (data["flux_err_min"]))) +
                0.100000*np.tanh(((((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) - ((((data["4__skewness_y"]) > (data["flux_dif2"]))*1.)))) - (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(np.minimum(((data["distmod"])), ((data["distmod"]))) > -1, data["flux_d0_pb5"], (((data["distmod"]) + ((((data["distmod"]) + (data["distmod"]))/2.0)))/2.0) )) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, data["5__fft_coefficient__coeff_1__attr__abs__y"], ((data["distmod"]) * 2.0) )) +
                0.100000*np.tanh(((((np.minimum(((((data["hostgal_photoz_err"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"])))), ((data["detected_flux_ratio_sq_skew"])))) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)) +
                0.100000*np.tanh((((((data["flux_d0_pb5"]) + (data["flux_d1_pb5"]))/2.0)) + (data["distmod"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((((data["5__skewness_x"]) + (data["hostgal_photoz_err"])))), ((data["2__kurtosis_x"]))))), ((((((data["flux_min"]) + (data["2__kurtosis_x"]))) + (0.367879)))))) +
                0.100000*np.tanh(((((data["hostgal_photoz_err"]) - (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) - (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))) +
                0.100000*np.tanh(((((((((data["detected_flux_min"]) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) * 2.0)) + (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) + (((data["detected_flux_min"]) - (data["detected_mjd_diff"]))))) +
                0.100000*np.tanh((((np.minimum(((((data["2__kurtosis_x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"])))), ((((data["4__kurtosis_x"]) + ((((1.0) + (data["distmod"]))/2.0))))))) + (data["flux_d0_pb5"]))/2.0)) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_median"])), ((((np.minimum(((((((data["1__kurtosis_x"]) - (data["1__skewness_y"]))) + (data["flux_d0_pb5"])))), ((data["flux_d1_pb5"])))) - (data["1__skewness_y"])))))) +
                0.100000*np.tanh(((((data["1__kurtosis_x"]) + (np.maximum(((((data["1__kurtosis_x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["hostgal_photoz"])))))) - (((data["2__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0)))) +
                0.100000*np.tanh(((((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["detected_mjd_diff"]) - (data["5__skewness_x"]))))) - (data["detected_mjd_diff"]))) - (data["flux_max"]))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["3__skewness_x"]) - (((((data["detected_flux_err_std"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["5__kurtosis_y"]))))))) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(3.141593 > -1, data["hostgal_photoz_err"], np.minimum(((data["detected_flux_min"])), (((0.28365617990493774)))) )) +
                0.100000*np.tanh(((data["1__kurtosis_x"]) + (((data["flux_d1_pb2"]) * 2.0)))) +
                0.100000*np.tanh(((data["flux_d1_pb5"]) - (np.where(data["flux_d0_pb2"]<0, data["1__skewness_y"], data["2__kurtosis_x"] )))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) +
                0.100000*np.tanh(((((data["mjd_size"]) + (((data["distmod"]) - (((data["detected_flux_by_flux_ratio_sq_sum"]) + (data["1__skewness_y"]))))))) - (data["3__skewness_y"]))) +
                0.100000*np.tanh(np.where(data["2__skewness_x"]<0, data["2__skewness_x"], ((((data["2__skewness_x"]) * 2.0)) * 2.0) )) +
                0.100000*np.tanh((-1.0*(((-1.0*(((((((-1.0*((data["mjd_diff"])))) + (data["flux_d0_pb4"]))) - (data["2__skewness_y"]))))))))) +
                0.100000*np.tanh(((((((data["flux_d0_pb5"]) - (data["flux_d1_pb0"]))) - (data["detected_mjd_diff"]))) - (((data["flux_d0_pb5"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(((((((((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_mjd_diff"]))) * 2.0)) - (((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))))) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(((((((data["detected_flux_dif3"]) - ((((data["flux_d1_pb5"]) < (data["2__fft_coefficient__coeff_0__attr__abs__x"]))*1.)))) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) - (data["3__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["1__kurtosis_x"]) + (data["3__skewness_x"]))) + (data["distmod"]))) +
                0.100000*np.tanh(((((np.where(data["5__skewness_x"] > -1, data["5__fft_coefficient__coeff_1__attr__abs__x"], ((data["flux_d0_pb5"]) + (data["5__skewness_x"])) )) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) +
                0.100000*np.tanh((((((data["1__skewness_y"]) < (((data["2__kurtosis_x"]) + (data["detected_mjd_diff"]))))*1.)) - ((((data["detected_mjd_diff"]) > ((((data["detected_mjd_diff"]) > (data["detected_mjd_diff"]))*1.)))*1.)))) +
                0.100000*np.tanh(((((((data["3__kurtosis_x"]) + (data["3__kurtosis_x"]))) + (data["2__kurtosis_x"]))) + (2.718282))) +
                0.100000*np.tanh(np.where(-2.0 > -1, ((data["flux_dif3"]) - (((data["1__skewness_x"]) - (data["detected_flux_by_flux_ratio_sq_skew"])))), ((((data["1__skewness_x"]) - (data["5__kurtosis_y"]))) * 2.0) )) +
                0.100000*np.tanh(((((((((data["detected_flux_dif3"]) * 2.0)) - (data["detected_mjd_diff"]))) * 2.0)) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((data["hostgal_photoz_err"]) - ((-1.0*((data["distmod"])))))) +
                0.100000*np.tanh(((((np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"]<0, data["5__fft_coefficient__coeff_1__attr__abs__y"], data["5__fft_coefficient__coeff_1__attr__abs__y"] )) - (((2.0) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))))) - (data["detected_flux_by_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh((((((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (np.maximum(((data["0__fft_coefficient__coeff_0__attr__abs__y"])), ((data["1__skewness_y"])))))) < ((((data["1__skewness_y"]) < (data["0__fft_coefficient__coeff_0__attr__abs__y"]))*1.)))*1.)) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["5__skewness_x"]) * 2.0)) - (data["flux_d1_pb0"]))) +
                0.100000*np.tanh(((np.maximum(((data["mjd_diff"])), ((data["2__skewness_x"])))) - (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh((((((((data["detected_mjd_diff"]) * 2.0)) < (data["flux_d0_pb5"]))*1.)) - (data["flux_d1_pb0"]))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) - (data["flux_ratio_sq_sum"]))) - (((((data["3__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))))) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) + (((data["2__skewness_x"]) + (((((((np.tanh((data["flux_ratio_sq_sum"]))) / 2.0)) * 2.0)) * 2.0)))))) +
                0.100000*np.tanh((((((data["flux_w_mean"]) < (data["detected_mjd_diff"]))*1.)) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((((((np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["5__fft_coefficient__coeff_1__attr__abs__x"], ((data["flux_d0_pb5"]) * 2.0) )) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((((data["1__skewness_x"]) - (data["1__skewness_y"]))) + (data["1__kurtosis_x"]))) - (data["1__skewness_y"]))) +
                0.100000*np.tanh(((((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0)) * (data["detected_flux_dif3"]))) + (np.minimum(((data["3__kurtosis_x"])), ((data["3__kurtosis_x"])))))) +
                0.100000*np.tanh(((((((data["distmod"]) + (data["flux_dif2"]))/2.0)) + (data["flux_dif2"]))/2.0)) +
                0.100000*np.tanh((((((data["flux_d0_pb2"]) < (data["flux_mean"]))*1.)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["2__fft_coefficient__coeff_1__attr__abs__y"])), ((data["flux_d0_pb0"])))) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_dif3"]) + (((data["detected_flux_dif3"]) + (((data["detected_flux_dif3"]) - (((data["flux_max"]) * 2.0)))))))) +
                0.100000*np.tanh(((((data["0__kurtosis_y"]) + (((data["3__kurtosis_y"]) + (data["detected_flux_err_min"]))))) + ((-1.0*((data["4__skewness_y"])))))) +
                0.100000*np.tanh(((((data["flux_err_min"]) + (data["2__skewness_x"]))) + (((((data["2__skewness_x"]) * 2.0)) + (np.where(data["2__skewness_x"]<0, data["2__skewness_x"], data["2__skewness_x"] )))))) +
                0.100000*np.tanh(np.where(data["detected_flux_err_min"] > -1, np.tanh((data["detected_flux_dif3"])), data["detected_flux_dif3"] )) +
                0.100000*np.tanh(np.where(np.where(np.where(data["4__kurtosis_x"]<0, data["detected_flux_dif3"], data["flux_d0_pb0"] )<0, data["flux_d0_pb0"], data["mwebv"] ) > -1, data["flux_d0_pb0"], ((data["flux_d0_pb0"]) * 2.0) )) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"]>0, ((data["flux_d0_pb5"]) + (data["detected_flux_err_max"])), np.where(data["5__skewness_x"]<0, data["detected_flux_dif3"], ((data["detected_flux_err_min"]) + (data["0__kurtosis_y"])) ) )) +
                0.100000*np.tanh(((np.where(((data["mjd_size"]) - (((data["mwebv"]) - (data["5__kurtosis_y"]))))<0, data["3__kurtosis_y"], data["3__kurtosis_y"] )) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["1__skewness_x"])), ((((data["detected_flux_dif3"]) * (data["detected_flux_dif3"])))))) * (data["detected_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh((((((np.where(((data["flux_d0_pb1"]) - (data["flux_median"])) > -1, data["flux_d1_pb1"], data["flux_d0_pb1"] )) > (data["flux_d0_pb1"]))*1.)) - (data["flux_median"]))) +
                0.100000*np.tanh(((((((((((((data["detected_flux_dif3"]) + (data["flux_dif3"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) + (data["flux_dif3"]))) +
                0.100000*np.tanh(((((((np.where(((data["detected_flux_max"]) * 2.0)>0, ((data["flux_d1_pb4"]) - (data["flux_d0_pb1"])), data["detected_flux_err_min"] )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) + (data["2__fft_coefficient__coeff_1__attr__abs__y"])))), ((data["flux_d1_pb2"])))) +
                0.100000*np.tanh((((((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) > (np.where(data["2__fft_coefficient__coeff_0__attr__abs__y"]>0, data["4__fft_coefficient__coeff_1__attr__abs__x"], data["5__fft_coefficient__coeff_1__attr__abs__x"] )))*1.)) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]>0, np.where(data["1__skewness_x"]>0, data["5__fft_coefficient__coeff_1__attr__abs__x"], data["3__kurtosis_x"] ), data["5__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(((((((data["detected_flux_dif3"]) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["flux_median"]))) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_err_min"]) + (np.maximum(((((((data["detected_flux_err_min"]) + (data["hostgal_photoz"]))) + (np.maximum(((data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((data["detected_flux_err_min"]))))))), ((data["5__fft_coefficient__coeff_1__attr__abs__y"])))))) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, (((data["flux_diff"]) < (data["1__skewness_x"]))*1.), data["detected_flux_by_flux_ratio_sq_sum"] )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb3"]>0, data["4__kurtosis_x"], np.where(data["flux_d1_pb4"]>0, np.where(data["flux_d1_pb3"]>0, data["5__fft_coefficient__coeff_1__attr__abs__x"], data["flux_err_skew"] ), data["4__fft_coefficient__coeff_0__attr__abs__y"] ) )) +
                0.100000*np.tanh(np.where(np.where(data["distmod"]<0, data["distmod"], data["mwebv"] )<0, ((data["detected_flux_dif3"]) + (data["0__kurtosis_y"])), data["distmod"] )) +
                0.100000*np.tanh((((data["distmod"]) + (np.where(data["hostgal_photoz"]>0, (((data["distmod"]) < (data["5__fft_coefficient__coeff_1__attr__abs__x"]))*1.), data["mwebv"] )))/2.0)) +
                0.100000*np.tanh(np.where(data["flux_d1_pb1"]<0, data["detected_flux_err_median"], data["flux_max"] )) +
                0.100000*np.tanh(((((data["0__skewness_x"]) - (data["2__fft_coefficient__coeff_0__attr__abs__y"]))) * (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.where(data["2__skewness_x"] > -1, data["detected_flux_dif3"], data["detected_mjd_size"] )) +
                0.100000*np.tanh(np.where(data["detected_flux_dif3"] > -1, ((data["2__fft_coefficient__coeff_1__attr__abs__y"]) * (data["detected_flux_ratio_sq_skew"])), ((np.where(data["flux_mean"]<0, data["flux_ratio_sq_skew"], data["5__fft_coefficient__coeff_1__attr__abs__x"] )) * (data["3__fft_coefficient__coeff_1__attr__abs__x"])) )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"]>0, (((data["mjd_size"]) < (data["hostgal_photoz"]))*1.), (((data["mwebv"]) + (np.minimum(((data["hostgal_photoz_err"])), ((data["1__skewness_x"])))))/2.0) )) +
                0.100000*np.tanh(((((((data["flux_d1_pb3"]) * 2.0)) * (np.where(np.where(data["flux_d1_pb3"] > -1, data["4__fft_coefficient__coeff_0__attr__abs__y"], data["flux_d1_pb4"] ) > -1, data["4__fft_coefficient__coeff_0__attr__abs__y"], data["flux_d1_pb3"] )))) * 2.0)) +
                0.100000*np.tanh(np.maximum(((data["flux_d0_pb0"])), ((((data["flux_d0_pb0"]) * 2.0))))) +
                0.100000*np.tanh(((((((((data["0__kurtosis_x"]) - (data["detected_mean"]))) - (((data["mjd_size"]) * 2.0)))) - (((data["0__kurtosis_x"]) * 2.0)))) - (data["5__kurtosis_y"]))) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"]<0, data["flux_d0_pb0"], (((data["4__fft_coefficient__coeff_1__attr__abs__y"]) > (np.where((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) > (data["3__kurtosis_x"]))*1.) > -1, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["3__kurtosis_x"] )))*1.) )) +
                0.100000*np.tanh(((((((((((np.where(data["detected_mean"]>0, data["detected_flux_median"], data["2__skewness_y"] )) - (data["flux_median"]))) * 2.0)) * 2.0)) - (data["flux_median"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"]<0, data["0__fft_coefficient__coeff_1__attr__abs__x"], np.where(data["flux_d0_pb5"]<0, ((data["distmod"]) + (data["3__fft_coefficient__coeff_0__attr__abs__x"])), data["detected_flux_dif3"] ) )) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_err_median"]))) + (((data["detected_flux_err_median"]) + (((data["2__skewness_x"]) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))))) +
                0.100000*np.tanh(np.where(data["distmod"]<0, np.where(data["distmod"] > -1, (-1.0*((data["flux_median"]))), data["distmod"] ), ((data["hostgal_photoz_err"]) + (data["flux_median"])) )) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb0"]<0, data["2__skewness_x"], np.where(data["flux_d0_pb0"]<0, data["flux_ratio_sq_skew"], (7.80123519897460938) ) )) * 2.0)) +
                0.100000*np.tanh(((((data["4__kurtosis_x"]) * (data["4__kurtosis_x"]))) * (((data["flux_d0_pb5"]) * (data["1__skewness_x"]))))) +
                0.100000*np.tanh(np.where(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))<0, ((data["mwebv"]) - (data["3__kurtosis_x"])), np.where(data["distmod"]<0, data["3__fft_coefficient__coeff_1__attr__abs__x"], data["5__fft_coefficient__coeff_1__attr__abs__x"] ) )) +
                0.100000*np.tanh(((((data["detected_flux_dif3"]) - (data["flux_median"]))) + (((data["detected_flux_dif3"]) - (((data["detected_flux_dif3"]) * (((data["detected_flux_dif3"]) - (data["flux_median"]))))))))) +
                0.100000*np.tanh(((np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["2__fft_coefficient__coeff_1__attr__abs__y"], ((((np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["2__fft_coefficient__coeff_1__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] )) * 2.0)) - (data["2__fft_coefficient__coeff_1__attr__abs__y"])) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["hostgal_photoz_err"]<0, np.where(data["hostgal_photoz"]<0, np.where(data["detected_flux_err_skew"]<0, data["hostgal_photoz_err"], data["hostgal_photoz"] ), data["hostgal_photoz_err"] ), data["hostgal_photoz_err"] )) +
                0.100000*np.tanh(np.where(np.where(data["flux_skew"]>0, data["1__fft_coefficient__coeff_1__attr__abs__y"], data["flux_skew"] )>0, data["distmod"], ((data["flux_skew"]) + (data["0__skewness_x"])) )) +
                0.100000*np.tanh(np.where(data["flux_ratio_sq_skew"]<0, data["detected_flux_err_min"], data["3__kurtosis_y"] )) +
                0.100000*np.tanh(((((np.where(data["hostgal_photoz_err"]>0, data["distmod"], ((((data["hostgal_photoz_err"]) + (data["flux_d1_pb4"]))) + (data["flux_d1_pb5"])) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(data["detected_flux_err_median"]>0, data["flux_err_min"], ((np.where(data["detected_flux_err_median"]>0, data["flux_err_min"], data["2__skewness_x"] )) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(np.where(((data["1__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0)<0, data["5__fft_coefficient__coeff_1__attr__abs__x"], (((data["1__fft_coefficient__coeff_0__attr__abs__x"]) < (data["5__fft_coefficient__coeff_1__attr__abs__x"]))*1.) )) +
                0.100000*np.tanh(((data["detected_flux_err_max"]) - (data["0__skewness_y"]))) +
                0.100000*np.tanh((((((data["detected_flux_err_std"]) < ((((data["detected_mjd_size"]) > (((data["flux_by_flux_ratio_sq_sum"]) - (np.maximum(((data["flux_mean"])), ((data["detected_flux_err_skew"])))))))*1.)))*1.)) - (data["detected_flux_err_std"]))) +
                0.100000*np.tanh((((data["detected_mjd_diff"]) < (((((((data["detected_mjd_size"]) < ((((data["2__kurtosis_y"]) < ((((data["detected_mjd_diff"]) < (data["detected_flux_min"]))*1.)))*1.)))*1.)) < (data["4__fft_coefficient__coeff_0__attr__abs__x"]))*1.)))*1.)) +
                0.100000*np.tanh(np.where(data["flux_err_skew"]>0, ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (((((data["2__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0)) - (data["1__skewness_x"])))), data["2__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(np.where(data["flux_dif2"]>0, np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"]>0, data["3__kurtosis_y"], data["0__fft_coefficient__coeff_1__attr__abs__x"] ), data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"] > -1, (((data["5__fft_coefficient__coeff_1__attr__abs__x"]) > (data["1__fft_coefficient__coeff_1__attr__abs__x"]))*1.), data["5__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.minimum(((data["4__kurtosis_x"])), ((data["3__skewness_x"])))) +
                0.100000*np.tanh(np.where(np.where(data["0__skewness_x"] > -1, (((data["flux_d0_pb2"]) < (data["flux_d0_pb2"]))*1.), data["flux_d0_pb2"] )<0, data["detected_mjd_size"], (((data["flux_d0_pb2"]) < (data["flux_mean"]))*1.) )) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["flux_diff"])), ((data["2__skewness_x"]))))), ((np.minimum(((np.minimum(((data["flux_d1_pb1"])), ((data["distmod"]))))), ((data["flux_d1_pb1"]))))))) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"] > -1, (((data["5__skewness_x"]) < (data["5__fft_coefficient__coeff_1__attr__abs__y"]))*1.), data["detected_mjd_size"] )) +
                0.100000*np.tanh(np.where(data["flux_ratio_sq_skew"]>0, np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]>0, data["flux_err_skew"], np.where(data["detected_flux_err_median"]>0, data["flux_ratio_sq_skew"], data["flux_ratio_sq_skew"] ) ), data["detected_flux_err_min"] )) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb4"]>0, np.where(data["hostgal_photoz_err"]<0, np.where(data["flux_d0_pb4"]<0, data["hostgal_photoz_err"], data["detected_mjd_size"] ), data["detected_mjd_size"] ), data["hostgal_photoz_err"] )) * 2.0)) +
                0.100000*np.tanh((((data["detected_mjd_diff"]) < ((((((data["detected_mjd_diff"]) < (data["detected_mjd_diff"]))*1.)) * 2.0)))*1.)) +
                0.100000*np.tanh(np.where(((data["flux_dif2"]) * (data["flux_dif2"]))>0, data["1__fft_coefficient__coeff_0__attr__abs__y"], np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]>0, data["1__fft_coefficient__coeff_0__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] ) )))

    def GP_class_64(self,data):
        return (-2.164979 +
                0.100000*np.tanh(((((((((((data["flux_by_flux_ratio_sq_skew"]) * 2.0)) + (((((((data["flux_by_flux_ratio_sq_skew"]) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - (((((((data["detected_mjd_diff"]) * 2.0)) * 2.0)) * 2.0)))) - (((data["detected_flux_by_flux_ratio_sq_sum"]) - (data["detected_mjd_diff"]))))) +
                0.100000*np.tanh((((-1.0*((((data["detected_mjd_diff"]) * 2.0))))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((data["flux_err_max"]) - (np.where(((((data["detected_mjd_diff"]) * 2.0)) * 2.0) > -1, data["detected_mjd_diff"], data["detected_mjd_diff"] )))) - (np.tanh((data["detected_mjd_diff"]))))) +
                0.100000*np.tanh((((((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"]))/2.0)) + ((-1.0*((data["detected_mjd_size"])))))) +
                0.100000*np.tanh(((((((((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - (((((data["detected_mjd_diff"]) * 2.0)) * 2.0)))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))) - (((data["detected_mjd_diff"]) * 2.0)))) +
                0.100000*np.tanh((((((-1.0*((np.where((-1.0*((data["detected_mjd_diff"])))>0, data["detected_mjd_diff"], data["detected_mjd_diff"] ))))) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(((((((((np.tanh((data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) - (((data["detected_mjd_diff"]) / 2.0)))) +
                0.100000*np.tanh(((((np.maximum(((data["flux_ratio_sq_skew"])), ((data["flux_ratio_sq_skew"])))) * 2.0)) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh((((((((data["detected_mjd_diff"]) > (data["detected_mjd_diff"]))*1.)) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) - (((((data["detected_mjd_diff"]) * 2.0)) * 2.0)))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_size"]))) +
                0.100000*np.tanh((((((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) > (data["1__skewness_y"]))*1.)) - ((((data["detected_mjd_diff"]) + (data["detected_mjd_diff"]))/2.0)))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((np.where(((data["flux_ratio_sq_skew"]) * 2.0) > -1, (-1.0*((data["detected_mjd_diff"]))), ((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"])) )) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_ratio_sq_skew"]) + (((data["flux_ratio_sq_skew"]) * 2.0)))) + (data["flux_ratio_sq_skew"]))) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((np.minimum((((-1.0*((data["detected_mjd_diff"]))))), (((((((-1.0*((data["detected_mjd_diff"])))) * 2.0)) + (data["flux_ratio_sq_skew"])))))) * 2.0)) +
                0.100000*np.tanh(((((((data["detected_flux_mean"]) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((((((data["flux_ratio_sq_skew"]) - (data["detected_mjd_size"]))) - (((data["detected_mjd_size"]) - (data["flux_d0_pb1"]))))) - (data["detected_mjd_size"]))) * 2.0)) +
                0.100000*np.tanh((-1.0*((((1.0) + (((data["detected_mjd_diff"]) + (data["detected_mjd_diff"])))))))) +
                0.100000*np.tanh((-1.0*((((((((1.0) + (data["detected_mjd_diff"]))) * 2.0)) + (((((data["detected_mjd_diff"]) * 2.0)) + (1.0)))))))) +
                0.100000*np.tanh((-1.0*((((((((data["detected_mjd_diff"]) * 2.0)) * 2.0)) - (((data["detected_mjd_diff"]) * 2.0))))))) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) + (((data["flux_ratio_sq_skew"]) * 2.0)))) + (((((data["flux_ratio_sq_skew"]) - (data["flux_by_flux_ratio_sq_sum"]))) * 2.0)))) +
                0.100000*np.tanh(((((np.minimum(((data["flux_ratio_sq_skew"])), ((((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"])))))) + (data["flux_ratio_sq_skew"]))) * 2.0)) +
                0.100000*np.tanh(((((((data["4__fft_coefficient__coeff_1__attr__abs__y"]) - (data["detected_mjd_size"]))) - (data["detected_mjd_size"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh((-1.0*((np.where((((((((data["detected_mjd_diff"]) / 2.0)) + (data["flux_median"]))/2.0)) * 2.0)<0, data["detected_mjd_diff"], ((data["detected_mjd_diff"]) * 2.0) ))))) +
                0.100000*np.tanh((-1.0*((np.where(data["detected_mjd_diff"] > -1, np.where(((data["detected_mjd_diff"]) / 2.0) > -1, 2.718282, data["detected_mjd_diff"] ), data["detected_mjd_diff"] ))))) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, -3.0, np.where(data["detected_mjd_diff"] > -1, -3.0, (8.0) ) )) +
                0.100000*np.tanh(((((((data["flux_ratio_sq_skew"]) - (data["flux_ratio_sq_skew"]))) * 2.0)) + (((np.maximum(((data["detected_flux_std"])), ((data["detected_flux_std"])))) + (data["flux_ratio_sq_skew"]))))) +
                0.100000*np.tanh(((((((data["detected_flux_std"]) - (data["detected_flux_diff"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"]))) + (((data["detected_flux_dif2"]) * 2.0)))) - (data["detected_mjd_size"]))) +
                0.100000*np.tanh(((((((-1.0) * 2.0)) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((((((data["detected_flux_std"]) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["flux_d0_pb5"]))) * 2.0)) +
                0.100000*np.tanh(((((data["detected_flux_std"]) - (data["flux_d0_pb4"]))) - (data["ddf"]))) +
                0.100000*np.tanh(((((((data["detected_flux_std"]) - (data["flux_d0_pb4"]))) - (data["flux_d0_pb4"]))) * 2.0)) +
                0.100000*np.tanh(((((((((data["flux_max"]) - (data["flux_d0_pb5"]))) - (((data["flux_d0_pb5"]) + (data["detected_mjd_diff"]))))) - (data["flux_median"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((data["flux_err_mean"]) + (((((((data["detected_flux_std"]) - (data["detected_mjd_diff"]))) - (data["flux_median"]))) - (data["flux_median"]))))) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((((np.minimum(((data["detected_flux_std"])), ((((data["flux_dif2"]) + (data["detected_flux_mean"])))))) - (data["flux_d0_pb5"]))) * 2.0)) +
                0.100000*np.tanh(((((data["detected_flux_std"]) + (((data["detected_flux_std"]) + (-2.0))))) + (((data["detected_flux_std"]) - (data["detected_mjd_diff"]))))) +
                0.100000*np.tanh(np.where((-1.0*((data["hostgal_photoz_err"]))) > -1, np.where((2.44751977920532227) > -1, np.where(data["detected_mjd_diff"] > -1, -3.0, 3.0 ), 3.0 ), 3.0 )) +
                0.100000*np.tanh((((-1.0*((((data["flux_d0_pb5"]) + (((data["flux_d0_pb5"]) + (data["flux_d0_pb4"])))))))) - (data["flux_d0_pb5"]))) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, -2.0, (-1.0*((-2.0))) )) +
                0.100000*np.tanh(((((data["detected_flux_std"]) + (((data["flux_err_max"]) + (np.tanh(((-1.0*((data["flux_dif3"])))))))))) * 2.0)) +
                0.100000*np.tanh(((((data["detected_flux_std"]) * 2.0)) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["flux_err_mean"]) - (np.where(data["detected_mjd_diff"]<0, data["detected_mjd_diff"], data["flux_err_mean"] )))) - (np.where(data["flux_err_mean"]<0, data["detected_mjd_diff"], (8.0) )))) +
                0.100000*np.tanh((((((data["hostgal_photoz"]) + (data["flux_dif2"]))/2.0)) + (((data["detected_flux_diff"]) + (data["3__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(((np.where(data["detected_mjd_diff"] > -1, -3.0, data["detected_flux_median"] )) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((((((data["detected_flux_diff"]) - (data["0__kurtosis_x"]))) - (data["1__kurtosis_x"]))) - (data["0__kurtosis_x"]))) - (data["flux_d0_pb5"]))) +
                0.100000*np.tanh(((((((np.minimum(((data["detected_flux_diff"])), ((data["detected_flux_diff"])))) - (data["1__skewness_x"]))) - (data["1__skewness_x"]))) - ((2.0)))) +
                0.100000*np.tanh(((((((data["detected_flux_diff"]) - (((1.0) - (((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))))))) - (data["mjd_size"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((data["detected_flux_std"]) - (np.where(data["5__fft_coefficient__coeff_0__attr__abs__y"]<0, np.where(data["detected_flux_std"] > -1, data["5__skewness_x"], data["detected_flux_std"] ), ((data["3__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_std"])) )))) +
                0.100000*np.tanh(np.where(((data["hostgal_photoz"]) + (data["flux_d0_pb4"])) > -1, ((data["hostgal_photoz"]) - (3.141593)), (-1.0*((data["hostgal_photoz"]))) )) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_std"])), ((data["detected_flux_std"])))) +
                0.100000*np.tanh(((np.where(data["flux_ratio_sq_skew"] > -1, data["flux_ratio_sq_skew"], data["flux_ratio_sq_skew"] )) + (np.where(data["flux_ratio_sq_skew"] > -1, data["flux_ratio_sq_skew"], data["flux_err_skew"] )))) +
                0.100000*np.tanh(((((((((np.maximum(((data["hostgal_photoz"])), ((data["hostgal_photoz"])))) * 2.0)) * (((data["detected_mjd_diff"]) + (data["hostgal_photoz"]))))) - (2.718282))) * 2.0)) +
                0.100000*np.tanh(np.where(((data["flux_skew"]) * 2.0) > -1, ((data["hostgal_photoz"]) + (-3.0)), data["detected_flux_std"] )) +
                0.100000*np.tanh(((np.where(data["detected_flux_std"] > -1, ((data["detected_mean"]) - (data["flux_d0_pb5"])), data["flux_w_mean"] )) + (data["detected_flux_std"]))) +
                0.100000*np.tanh(np.where(data["flux_err_mean"]>0, -2.0, ((data["flux_err_mean"]) - (-2.0)) )) +
                0.100000*np.tanh(np.where(data["flux_median"]<0, data["flux_ratio_sq_skew"], ((data["detected_mjd_diff"]) * (-3.0)) )) +
                0.100000*np.tanh(((np.where(data["0__kurtosis_x"]<0, data["4__kurtosis_x"], ((data["hostgal_photoz"]) - (np.where(data["0__kurtosis_x"]<0, data["detected_flux_skew"], data["0__kurtosis_x"] ))) )) + (data["flux_d1_pb3"]))) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"]>0, ((data["4__fft_coefficient__coeff_0__attr__abs__x"]) + (np.minimum(((-3.0)), ((data["2__skewness_y"]))))), data["detected_flux_mean"] )) +
                0.100000*np.tanh(np.where(((data["distmod"]) + (data["distmod"])) > -1, ((data["hostgal_photoz"]) * 2.0), ((((data["hostgal_photoz"]) * 2.0)) + (3.0)) )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, ((-2.0) + (data["hostgal_photoz"])), data["flux_w_mean"] )) +
                0.100000*np.tanh(((((np.minimum(((((((data["detected_flux_std"]) - (data["flux_d1_pb5"]))) - (data["3__fft_coefficient__coeff_0__attr__abs__x"])))), ((data["detected_flux_w_mean"])))) - (data["3__fft_coefficient__coeff_0__attr__abs__x"]))) - (data["flux_d1_pb5"]))) +
                0.100000*np.tanh(((((((((data["detected_flux_skew"]) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) - (((data["0__fft_coefficient__coeff_0__attr__abs__x"]) - (data["0__skewness_y"]))))) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_std"]) + (((((data["2__skewness_x"]) + (data["2__kurtosis_x"]))) + (data["detected_flux_std"]))))) +
                0.100000*np.tanh((((data["hostgal_photoz"]) + (((((data["hostgal_photoz"]) - (2.0))) + (((data["hostgal_photoz"]) * (((data["hostgal_photoz"]) - ((3.13035678863525391)))))))))/2.0)) +
                0.100000*np.tanh((((((-3.0) + (((data["hostgal_photoz"]) + (data["detected_flux_diff"]))))/2.0)) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(((np.where(data["flux_skew"]<0, data["detected_flux_median"], data["flux_max"] )) + (data["flux_dif2"]))) +
                0.100000*np.tanh(((((-1.0) - (data["detected_mjd_diff"]))) - (((data["detected_mjd_diff"]) - (data["detected_flux_max"]))))) +
                0.100000*np.tanh((-1.0*(((-1.0*(((-1.0*((np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["mjd_diff"], data["detected_flux_err_skew"] ))))))))))) +
                0.100000*np.tanh(np.where(data["detected_flux_err_max"]>0, np.where(data["flux_d1_pb4"]>0, data["3__fft_coefficient__coeff_0__attr__abs__y"], data["4__fft_coefficient__coeff_0__attr__abs__y"] ), -2.0 )) +
                0.100000*np.tanh(np.where(((((1.0) - (data["flux_d0_pb5"]))) + (data["detected_flux_std"])) > -1, data["detected_flux_std"], data["ddf"] )) +
                0.100000*np.tanh(np.where(data["flux_err_std"]<0, data["detected_flux_std"], ((((data["flux_err_std"]) - (data["flux_err_std"]))) - (((data["detected_flux_std"]) + (data["detected_flux_std"])))) )) +
                0.100000*np.tanh((-1.0*((((data["3__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + ((((data["flux_d0_pb5"]) > ((-1.0*((data["mjd_size"])))))*1.))))))))) +
                0.100000*np.tanh(((data["flux_d1_pb0"]) + (((((data["flux_w_mean"]) - (data["0__kurtosis_x"]))) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))))) +
                0.100000*np.tanh(((((((((data["hostgal_photoz"]) - (data["distmod"]))) * 2.0)) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh(((((((data["hostgal_photoz"]) * 2.0)) * (np.where((1.0)<0, ((data["distmod"]) * 2.0), ((data["distmod"]) * 2.0) )))) + (-2.0))) +
                0.100000*np.tanh(np.where(data["0__kurtosis_x"]<0, np.where(data["3__fft_coefficient__coeff_0__attr__abs__x"]<0, data["flux_d1_pb4"], (((data["hostgal_photoz"]) + (data["hostgal_photoz"]))/2.0) ), ((data["2__fft_coefficient__coeff_1__attr__abs__y"]) + (data["hostgal_photoz"])) )) +
                0.100000*np.tanh(np.minimum(((data["2__skewness_x"])), ((np.minimum(((((data["3__fft_coefficient__coeff_0__attr__abs__x"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["2__skewness_x"]))))))) +
                0.100000*np.tanh(np.maximum(((data["2__skewness_x"])), ((((data["2__skewness_x"]) / 2.0))))) +
                0.100000*np.tanh(np.where(data["flux_median"]<0, np.where(data["flux_w_mean"] > -1, np.where(data["flux_w_mean"]<0, 2.0, data["detected_flux_max"] ), data["flux_w_mean"] ), -2.0 )) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, np.where(data["detected_mjd_diff"]>0, -2.0, -2.0 ), data["4__kurtosis_x"] )) +
                0.100000*np.tanh((((data["detected_flux_ratio_sq_skew"]) + (np.maximum((((((data["detected_flux_w_mean"]) + (data["flux_d1_pb3"]))/2.0))), ((np.maximum(((data["hostgal_photoz"])), ((data["hostgal_photoz"]))))))))/2.0)) +
                0.100000*np.tanh(np.tanh((((((np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"]>0, ((data["detected_flux_std"]) - (data["detected_flux_dif2"])), data["flux_w_mean"] )) * 2.0)) - (data["1__kurtosis_x"]))))) +
                0.100000*np.tanh(np.where(np.where(np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"]<0, data["0__fft_coefficient__coeff_1__attr__abs__x"], data["2__skewness_x"] )<0, data["hostgal_photoz"], data["flux_by_flux_ratio_sq_skew"] )<0, data["3__fft_coefficient__coeff_0__attr__abs__y"], data["mjd_diff"] )) +
                0.100000*np.tanh(((((((data["hostgal_photoz"]) + (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) + (data["detected_flux_err_skew"]))) + (data["detected_flux_err_skew"]))) +
                0.100000*np.tanh(np.where(data["flux_ratio_sq_skew"] > -1, np.where(np.maximum(((data["3__fft_coefficient__coeff_0__attr__abs__x"])), ((data["detected_flux_std"]))) > -1, data["detected_flux_skew"], data["detected_flux_err_std"] ), data["flux_dif2"] )) +
                0.100000*np.tanh((((-1.0*((((data["3__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0))))) - (((((data["hostgal_photoz_err"]) - ((((data["flux_diff"]) > (data["distmod"]))*1.)))) * (data["3__fft_coefficient__coeff_0__attr__abs__x"]))))) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]>0, data["3__fft_coefficient__coeff_0__attr__abs__y"], np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, data["detected_flux_err_mean"], np.where(data["detected_flux_err_max"]>0, data["3__fft_coefficient__coeff_0__attr__abs__y"], data["2__kurtosis_x"] ) ) )) +
                0.100000*np.tanh(((((data["detected_flux_dif2"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(((((((((data["ddf"]) - (data["detected_mjd_diff"]))) * 2.0)) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) + (data["ddf"]))) +
                0.100000*np.tanh((((np.minimum(((data["flux_std"])), ((np.tanh((data["flux_diff"])))))) + (((data["2__fft_coefficient__coeff_1__attr__abs__y"]) - (data["flux_d0_pb5"]))))/2.0)) +
                0.100000*np.tanh(((np.minimum(((data["flux_max"])), ((((data["5__skewness_y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"])))))) + (data["detected_flux_std"]))) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_0__attr__abs__y"]<0, data["3__fft_coefficient__coeff_0__attr__abs__y"], np.where(data["5__kurtosis_x"]<0, np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"]<0, data["2__skewness_x"], data["3__fft_coefficient__coeff_0__attr__abs__y"] ), data["3__fft_coefficient__coeff_0__attr__abs__y"] ) )) +
                0.100000*np.tanh(np.where(data["3__fft_coefficient__coeff_0__attr__abs__x"]<0, np.where(data["flux_d1_pb4"] > -1, data["flux_d1_pb4"], np.where(data["3__fft_coefficient__coeff_0__attr__abs__x"]<0, data["flux_d1_pb4"], data["flux_ratio_sq_skew"] ) ), data["2__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_0__attr__abs__x"]) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["detected_mjd_diff"], np.where(data["hostgal_photoz_err"] > -1, data["flux_diff"], data["detected_flux_err_mean"] ) )) +
                0.100000*np.tanh(((data["hostgal_photoz"]) - (3.0))) +
                0.100000*np.tanh(np.where(data["0__kurtosis_x"]<0, data["4__kurtosis_x"], data["2__kurtosis_x"] )) +
                0.100000*np.tanh(np.where(np.where(data["2__kurtosis_x"]<0, data["3__skewness_x"], data["2__skewness_x"] ) > -1, data["2__kurtosis_x"], np.where(data["2__kurtosis_x"] > -1, data["2__skewness_x"], data["3__skewness_x"] ) )) +
                0.100000*np.tanh(((((np.minimum(((((data["ddf"]) - (((data["flux_err_max"]) - (data["ddf"])))))), ((data["0__skewness_x"])))) - (data["ddf"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((((((np.minimum(((-2.0)), ((data["detected_mjd_diff"])))) - (data["distmod"]))) - (((data["distmod"]) * 2.0)))) * 2.0)) - (data["distmod"]))) +
                0.100000*np.tanh((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["hostgal_photoz"]))/2.0)) + (((data["0__kurtosis_y"]) + (((data["detected_flux_dif2"]) + (data["hostgal_photoz"]))))))) +
                0.100000*np.tanh((-1.0*((np.where(data["detected_mjd_diff"]<0, np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"]<0, data["detected_mjd_diff"], data["3__kurtosis_y"] ), np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["5__kurtosis_x"], data["detected_mjd_diff"] ) ))))) +
                0.100000*np.tanh(np.where(data["flux_by_flux_ratio_sq_sum"]<0, ((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["mjd_diff"]))) - (data["mjd_diff"]))) / 2.0), ((data["flux_err_std"]) - (data["mjd_diff"])) )) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, data["2__kurtosis_x"], np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, data["5__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, data["2__kurtosis_x"], data["3__fft_coefficient__coeff_0__attr__abs__y"] ) ) )) +
                0.100000*np.tanh(np.where(data["distmod"]<0, data["detected_flux_std"], data["detected_flux_err_mean"] )) +
                0.100000*np.tanh((((data["2__fft_coefficient__coeff_1__attr__abs__y"]) + (data["hostgal_photoz"]))/2.0)) +
                0.100000*np.tanh(((((data["detected_flux_diff"]) / 2.0)) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.tanh((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (data["1__fft_coefficient__coeff_0__attr__abs__x"]))))) +
                0.100000*np.tanh(np.where(data["5__kurtosis_y"]<0, (-1.0*((data["2__fft_coefficient__coeff_1__attr__abs__x"]))), np.tanh((np.where(((data["flux_dif2"]) * (data["1__fft_coefficient__coeff_1__attr__abs__x"])) > -1, data["flux_dif2"], data["2__fft_coefficient__coeff_1__attr__abs__x"] ))) )) +
                0.100000*np.tanh(np.where(data["4__kurtosis_x"]>0, data["detected_flux_w_mean"], ((data["detected_flux_dif2"]) / 2.0) )) +
                0.100000*np.tanh(np.where(data["3__fft_coefficient__coeff_1__attr__abs__y"]>0, data["3__fft_coefficient__coeff_1__attr__abs__y"], ((data["2__kurtosis_x"]) + (data["detected_flux_ratio_sq_sum"])) )) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["mjd_size"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"]))))), ((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_flux_dif2"])))))) +
                0.100000*np.tanh(((((-1.0*((data["flux_ratio_sq_sum"])))) + ((-1.0*((data["flux_ratio_sq_sum"])))))/2.0)) +
                0.100000*np.tanh(np.where(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * (data["3__fft_coefficient__coeff_1__attr__abs__y"]))>0, np.where(data["flux_d1_pb1"]<0, data["flux_err_mean"], data["2__fft_coefficient__coeff_0__attr__abs__x"] ), data["3__kurtosis_x"] )) +
                0.100000*np.tanh(np.where(data["5__kurtosis_x"] > -1, data["3__fft_coefficient__coeff_0__attr__abs__y"], (((data["3__fft_coefficient__coeff_1__attr__abs__y"]) + (data["4__skewness_y"]))/2.0) )) +
                0.100000*np.tanh(((data["hostgal_photoz"]) - (((((((((((((data["distmod"]) * 2.0)) - (data["distmod"]))) * 2.0)) - (data["hostgal_photoz"]))) * 2.0)) * 2.0)))) +
                0.100000*np.tanh(((((((data["distmod"]) * (np.where(data["3__fft_coefficient__coeff_0__attr__abs__x"]>0, data["flux_d1_pb4"], data["3__fft_coefficient__coeff_0__attr__abs__x"] )))) * 2.0)) * 2.0)))

    def GP_class_65(self,data):
        return (-0.972955 +
                0.100000*np.tanh((((-1.0*((((data["distmod"]) + (((data["distmod"]) + (((((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (data["4__fft_coefficient__coeff_0__attr__abs__y"]))) + (2.0)))))))))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((data["detected_mjd_diff"])), ((((np.minimum(((data["flux_ratio_sq_skew"])), ((((np.minimum(((data["detected_mjd_diff"])), ((data["flux_ratio_sq_skew"])))) * 2.0))))) * 2.0))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((-2.0) - (data["distmod"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, -3.0, np.where(data["distmod"] > -1, data["distmod"], ((data["flux_by_flux_ratio_sq_skew"]) * 2.0) ) )) +
                0.100000*np.tanh(((((-2.0) + (data["flux_ratio_sq_skew"]))) + (((((data["detected_mjd_size"]) - (data["detected_mjd_size"]))) - (data["distmod"]))))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - (((data["3__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)))) + (((np.minimum(((data["detected_mjd_diff"])), ((data["flux_by_flux_ratio_sq_skew"])))) - ((2.0)))))) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["detected_mjd_diff"], ((((data["detected_mjd_diff"]) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"])) )) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) - (((data["3__fft_coefficient__coeff_1__attr__abs__x"]) - (data["flux_ratio_sq_skew"]))))) - (data["detected_mjd_size"]))) +
                0.100000*np.tanh(((((((data["flux_by_flux_ratio_sq_skew"]) - (((data["4__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.minimum(((((data["detected_mjd_diff"]) * 2.0))), ((((((((((data["flux_ratio_sq_skew"]) - (data["detected_mjd_size"]))) * 2.0)) * 2.0)) - (data["detected_mjd_size"])))))) * 2.0)) +
                0.100000*np.tanh(np.where(np.where(data["hostgal_photoz"] > -1, data["flux_ratio_sq_skew"], data["hostgal_photoz"] ) > -1, np.where(data["4__fft_coefficient__coeff_1__attr__abs__x"] > -1, -2.0, data["hostgal_photoz"] ), data["flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(np.minimum(((np.where(((data["detected_mjd_diff"]) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))<0, -3.0, data["detected_mjd_diff"] ))), ((data["detected_mjd_diff"])))) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) - ((((((-3.0) < (data["4__fft_coefficient__coeff_1__attr__abs__y"]))*1.)) * 2.0)))) - (data["distmod"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((np.minimum((((((((data["flux_by_flux_ratio_sq_skew"]) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) + (-3.0))/2.0))), ((data["flux_by_flux_ratio_sq_skew"])))) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_ratio_sq_skew"]) + (data["detected_mjd_diff"]))) - (2.0))) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((np.minimum(((data["flux_ratio_sq_skew"])), ((((data["detected_flux_err_mean"]) - (data["5__fft_coefficient__coeff_1__attr__abs__y"])))))) + (np.minimum(((data["flux_ratio_sq_skew"])), ((data["2__kurtosis_x"])))))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))) + (((((data["flux_by_flux_ratio_sq_skew"]) * (data["detected_mjd_diff"]))) + (((-2.0) - (data["distmod"]))))))) +
                0.100000*np.tanh(((((data["flux_by_flux_ratio_sq_skew"]) - (((((((data["distmod"]) + ((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) + (data["4__fft_coefficient__coeff_1__attr__abs__y"]))/2.0)))/2.0)) + (3.141593))/2.0)))) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((data["flux_by_flux_ratio_sq_skew"]) + (((((((-2.0) - (data["distmod"]))) - (data["distmod"]))) + (((-3.0) + (data["flux_ratio_sq_skew"]))))))) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz"] > -1, np.where(data["flux_by_flux_ratio_sq_skew"] > -1, data["hostgal_photoz"], np.where(data["hostgal_photoz"] > -1, -2.0, data["flux_by_flux_ratio_sq_skew"] ) ), data["flux_by_flux_ratio_sq_skew"] )) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((((data["flux_by_flux_ratio_sq_skew"]) - (data["4__fft_coefficient__coeff_1__attr__abs__y"])))), ((data["detected_mjd_diff"])))) * 2.0)) + (np.tanh((((data["flux_ratio_sq_skew"]) * (data["detected_mjd_diff"]))))))) +
                0.100000*np.tanh(((((((((data["detected_mjd_diff"]) - (1.0))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) +
                0.100000*np.tanh(((((np.where(data["flux_ratio_sq_skew"] > -1, data["flux_ratio_sq_skew"], data["2__kurtosis_y"] )) - (((data["3__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, ((-2.0) - (data["flux_dif3"])), np.where(data["hostgal_photoz"] > -1, ((data["hostgal_photoz"]) - (data["hostgal_photoz"])), data["flux_by_flux_ratio_sq_skew"] ) )) +
                0.100000*np.tanh((((((data["flux_ratio_sq_skew"]) + (data["1__skewness_x"]))/2.0)) + ((((data["flux_ratio_sq_skew"]) + (((data["flux_skew"]) * 2.0)))/2.0)))) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, np.where(-3.0 > -1, data["hostgal_photoz"], -3.0 ), data["flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, -1.0, ((data["5__fft_coefficient__coeff_0__attr__abs__x"]) - (((data["4__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0))) )) +
                0.100000*np.tanh(np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((((((((data["detected_mjd_diff"]) + (data["detected_mjd_diff"]))) + (-3.0))) + (np.minimum(((data["flux_ratio_sq_skew"])), ((data["flux_ratio_sq_skew"]))))))))) +
                0.100000*np.tanh(((((np.minimum(((data["2__kurtosis_y"])), ((((-2.0) + (data["detected_mjd_diff"])))))) + (data["detected_mjd_diff"]))) - (data["detected_flux_by_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((((((data["1__skewness_x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) - (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["4__kurtosis_x"]) + (((data["flux_skew"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh((((((((data["flux_skew"]) - (data["distmod"]))) > (data["detected_flux_by_flux_ratio_sq_skew"]))*1.)) - (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))) +
                0.100000*np.tanh(np.where(-3.0<0, np.where(data["distmod"] > -1, -3.0, data["detected_mjd_diff"] ), np.where(data["detected_mjd_diff"] > -1, data["distmod"], data["flux_ratio_sq_skew"] ) )) +
                0.100000*np.tanh(((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["detected_mjd_diff"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))))) +
                0.100000*np.tanh(((((data["flux_by_flux_ratio_sq_skew"]) - (np.maximum(((data["2__skewness_y"])), (((((((data["2__skewness_x"]) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)) / 2.0))))))) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((-2.0) - (data["distmod"]))) - (((data["distmod"]) - (((-2.0) - (data["distmod"]))))))) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["0__skewness_x"]))) * 2.0)))) +
                0.100000*np.tanh(np.where(data["flux_skew"]>0, ((data["ddf"]) + (data["2__skewness_x"])), data["flux_d0_pb1"] )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, (((6.0)) - ((7.0))), (6.0) )) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + (((((np.minimum(((-1.0)), ((-1.0)))) + (data["detected_mjd_diff"]))) + (np.minimum(((0.0)), ((data["flux_by_flux_ratio_sq_sum"])))))))) +
                0.100000*np.tanh(np.where(np.minimum(((data["flux_ratio_sq_skew"])), ((data["flux_ratio_sq_skew"])))>0, np.where(data["flux_ratio_sq_skew"] > -1, data["flux_ratio_sq_skew"], data["flux_ratio_sq_skew"] ), data["3__kurtosis_x"] )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"]<0, np.where(data["hostgal_photoz"] > -1, np.where(data["flux_skew"] > -1, -3.0, data["flux_skew"] ), data["flux_skew"] ), data["flux_skew"] )) +
                0.100000*np.tanh((((-1.0*((data["distmod"])))) + (np.where(data["flux_median"]>0, -3.0, (((-3.0) + (data["3__kurtosis_x"]))/2.0) )))) +
                0.100000*np.tanh(((((data["3__kurtosis_x"]) - (((data["flux_d0_pb3"]) - (data["flux_skew"]))))) - (data["2__skewness_x"]))) +
                0.100000*np.tanh(np.where(((data["flux_by_flux_ratio_sq_skew"]) - (-3.0)) > -1, np.where(((data["flux_by_flux_ratio_sq_skew"]) + (data["distmod"])) > -1, -3.0, data["flux_ratio_sq_skew"] ), data["flux_by_flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(((((((((-2.0) + (data["distmod"]))) - (((data["distmod"]) - (-2.0))))) - (data["distmod"]))) - (data["distmod"]))) +
                0.100000*np.tanh(((data["1__kurtosis_y"]) + (((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"]))))) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) + (-2.0))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (((np.tanh((-2.0))) + (data["detected_mjd_diff"]))))) +
                0.100000*np.tanh((((((data["2__skewness_x"]) + (((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))/2.0)) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["2__skewness_x"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["detected_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) * (data["flux_skew"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) - ((((2.718282) > (data["detected_flux_ratio_sq_skew"]))*1.)))) +
                0.100000*np.tanh(((data["flux_std"]) + (np.where(data["3__kurtosis_x"]>0, data["detected_mjd_diff"], data["detected_mjd_diff"] )))) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) - (data["detected_flux_by_flux_ratio_sq_skew"]))) + (data["flux_skew"]))) +
                0.100000*np.tanh(np.where(data["3__fft_coefficient__coeff_1__attr__abs__y"] > -1, data["0__fft_coefficient__coeff_0__attr__abs__y"], (((data["3__kurtosis_x"]) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))/2.0) )) +
                0.100000*np.tanh((((((data["detected_flux_min"]) + (data["flux_ratio_sq_skew"]))/2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(-2.0 > -1, ((((-2.0) - (data["distmod"]))) * 2.0), ((((-2.0) - (data["distmod"]))) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(((np.where(data["flux_err_std"] > -1, ((((data["flux_err_std"]) * 2.0)) * 2.0), data["3__skewness_x"] )) * 2.0)) +
                0.100000*np.tanh(((data["2__kurtosis_y"]) + (((data["2__kurtosis_y"]) + (((data["3__skewness_x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))))) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["flux_median"]) - (data["flux_median"]))))) +
                0.100000*np.tanh(((((-2.0) + (((((np.minimum(((-2.0)), ((-2.0)))) - (data["distmod"]))) - (data["distmod"]))))) * 2.0)) +
                0.100000*np.tanh(((((((((((data["detected_flux_min"]) - (data["flux_mean"]))) * 2.0)) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["distmod"]))) - (2.0))) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["5__skewness_x"]) - (((data["distmod"]) + (3.0))))))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_by_flux_ratio_sq_skew"]))) * 2.0)) +
                0.100000*np.tanh(((((((((((data["flux_by_flux_ratio_sq_skew"]) - (data["flux_median"]))) - (data["flux_median"]))) - (data["flux_std"]))) - (data["flux_median"]))) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"] > -1, np.where(data["hostgal_photoz"] > -1, data["hostgal_photoz"], data["flux_skew"] ), np.where(data["hostgal_photoz"]>0, data["hostgal_photoz"], -3.0 ) )) +
                0.100000*np.tanh(np.where(((data["distmod"]) / 2.0) > -1, -3.0, ((data["distmod"]) + ((5.06640815734863281))) )) +
                0.100000*np.tanh(((((((((data["2__skewness_x"]) + (data["0__skewness_y"]))) - (data["detected_flux_by_flux_ratio_sq_skew"]))) - (data["flux_d0_pb1"]))) + (data["2__skewness_x"]))) +
                0.100000*np.tanh((((((((data["detected_flux_min"]) + (data["3__kurtosis_y"]))) + (data["5__kurtosis_y"]))) + (data["flux_skew"]))/2.0)) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"] > -1, data["0__fft_coefficient__coeff_0__attr__abs__y"], np.minimum(((data["flux_skew"])), ((data["0__skewness_x"]))) )) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - ((((((((data["flux_err_mean"]) * 2.0)) - (data["mwebv"]))) < (((((data["flux_err_std"]) / 2.0)) * (data["2__skewness_y"]))))*1.)))) +
                0.100000*np.tanh(np.where(data["0__skewness_x"] > -1, data["3__kurtosis_x"], data["3__kurtosis_x"] )) +
                0.100000*np.tanh(np.where(data["detected_flux_min"] > -1, data["flux_ratio_sq_skew"], np.minimum(((((data["flux_ratio_sq_skew"]) - (data["flux_by_flux_ratio_sq_skew"])))), ((((data["flux_ratio_sq_skew"]) - (data["flux_by_flux_ratio_sq_skew"]))))) )) +
                0.100000*np.tanh((((np.where((((data["1__kurtosis_y"]) < ((((data["3__kurtosis_x"]) + (data["0__skewness_x"]))/2.0)))*1.)<0, data["detected_flux_ratio_sq_sum"], data["1__kurtosis_x"] )) + (data["1__kurtosis_x"]))/2.0)) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_err_std"]))))) +
                0.100000*np.tanh(((np.where(((((((((data["flux_err_median"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)>0, data["3__kurtosis_x"], ((data["ddf"]) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(np.where(1.0<0, ((data["distmod"]) - (((-3.0) - (data["distmod"])))), ((-3.0) - (((data["distmod"]) * 2.0))) )) +
                0.100000*np.tanh(((((np.minimum(((-2.0)), ((((-2.0) - (data["distmod"])))))) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh(((data["flux_skew"]) + (np.where(data["flux_skew"]<0, data["flux_err_std"], (((data["1__skewness_x"]) + (np.where(data["0__kurtosis_y"] > -1, data["flux_d0_pb1"], data["flux_err_std"] )))/2.0) )))) +
                0.100000*np.tanh(((((((data["flux_by_flux_ratio_sq_skew"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) - (np.minimum((((-1.0*((data["3__skewness_x"]))))), ((((data["4__kurtosis_y"]) * 2.0))))))) +
                0.100000*np.tanh(((((((((data["0__skewness_x"]) + (data["0__skewness_x"]))/2.0)) + (data["mwebv"]))/2.0)) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((data["detected_flux_min"]) * 2.0))), ((((((data["flux_ratio_sq_skew"]) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["detected_flux_by_flux_ratio_sq_skew"])))))) +
                0.100000*np.tanh(((((((((data["detected_mjd_diff"]) * (data["detected_mjd_diff"]))) - (((data["flux_mean"]) - (data["detected_mjd_diff"]))))) - (data["detected_flux_err_skew"]))) - (1.0))) +
                0.100000*np.tanh(((data["2__skewness_x"]) + (data["2__skewness_x"]))) +
                0.100000*np.tanh(np.where(data["flux_diff"] > -1, data["4__kurtosis_x"], data["flux_err_mean"] )) +
                0.100000*np.tanh(((((((data["flux_median"]) - (data["flux_median"]))) - (np.maximum(((-2.0)), ((data["0__fft_coefficient__coeff_1__attr__abs__x"])))))) - (data["flux_std"]))) +
                0.100000*np.tanh(((((((((np.where(((data["distmod"]) / 2.0) > -1, -1.0, 0.367879 )) / 2.0)) * 2.0)) * 2.0)) / 2.0)) +
                0.100000*np.tanh(((((-2.0) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh((((((((((data["flux_d0_pb1"]) - (np.tanh(((7.0)))))) < (data["flux_w_mean"]))*1.)) + (data["flux_std"]))) - (data["flux_d0_pb1"]))) +
                0.100000*np.tanh(((data["0__skewness_x"]) + ((((((data["3__skewness_x"]) + ((((data["3__kurtosis_x"]) + (data["detected_flux_skew"]))/2.0)))/2.0)) + (-2.0))))) +
                0.100000*np.tanh(((((np.where(0.367879 > -1, ((data["ddf"]) + (((((data["flux_err_std"]) * 2.0)) * 2.0))), data["flux_err_std"] )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__y"])), ((data["0__fft_coefficient__coeff_0__attr__abs__y"])))) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_skew"]))))) +
                0.100000*np.tanh(((((data["flux_max"]) + (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh((((data["1__kurtosis_y"]) + (np.minimum(((((data["flux_ratio_sq_sum"]) + (np.where(data["2__kurtosis_y"] > -1, data["2__kurtosis_y"], data["3__kurtosis_x"] ))))), ((data["1__fft_coefficient__coeff_0__attr__abs__y"])))))/2.0)) +
                0.100000*np.tanh(((data["flux_ratio_sq_sum"]) * (np.tanh((data["detected_flux_ratio_sq_skew"]))))) +
                0.100000*np.tanh(((((((np.where(data["flux_err_std"] > -1, ((data["flux_err_std"]) * 2.0), data["flux_err_std"] )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["4__fft_coefficient__coeff_1__attr__abs__x"]) + (((data["detected_flux_skew"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"] > -1, np.maximum(((-3.0)), ((data["3__kurtosis_x"]))), data["hostgal_photoz"] )) +
                0.100000*np.tanh(np.maximum(((data["0__skewness_y"])), ((data["1__kurtosis_x"])))) +
                0.100000*np.tanh(((((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) / 2.0)) - (data["detected_flux_by_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh((((((data["2__kurtosis_x"]) - (data["detected_flux_by_flux_ratio_sq_skew"]))) + (((data["2__skewness_x"]) - (((data["detected_flux_by_flux_ratio_sq_skew"]) - (data["2__skewness_x"]))))))/2.0)) +
                0.100000*np.tanh(((((((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) / 2.0)) - (data["flux_d0_pb1"]))) * 2.0)) + (data["flux_skew"]))) +
                0.100000*np.tanh(((((data["flux_err_std"]) * 2.0)) * 2.0)) +
                0.100000*np.tanh((((data["flux_d1_pb5"]) > ((((data["flux_ratio_sq_sum"]) + (data["detected_mjd_diff"]))/2.0)))*1.)) +
                0.100000*np.tanh((((data["flux_err_std"]) + (data["flux_d1_pb5"]))/2.0)) +
                0.100000*np.tanh(((-2.0) + (np.maximum(((np.maximum(((-2.0)), ((((data["detected_mjd_diff"]) * (data["detected_mjd_diff"]))))))), ((data["detected_mjd_diff"])))))) +
                0.100000*np.tanh(((np.where(data["2__skewness_x"]<0, data["flux_err_std"], ((((((data["2__skewness_x"]) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)) + (data["detected_flux_w_mean"]))/2.0) )) * 2.0)) +
                0.100000*np.tanh((((((((data["0__skewness_x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) + (data["0__skewness_x"]))/2.0)) +
                0.100000*np.tanh((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) < (data["detected_flux_min"]))*1.)) +
                0.100000*np.tanh(((np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]>0, data["0__fft_coefficient__coeff_1__attr__abs__y"], data["detected_flux_ratio_sq_sum"] )) / 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_ratio_sq_sum"] > -1, data["1__skewness_x"], data["3__kurtosis_x"] )) +
                0.100000*np.tanh((((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + ((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["flux_skew"]) + (data["flux_ratio_sq_skew"]))))/2.0)))/2.0)) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"]>0, data["5__kurtosis_x"], np.where(np.minimum(((data["1__skewness_y"])), ((data["flux_err_std"])))>0, np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["5__kurtosis_x"]))), data["flux_err_std"] ) )) +
                0.100000*np.tanh(np.minimum(((np.minimum((((((data["detected_mjd_diff"]) + (np.minimum(((data["flux_mean"])), ((data["0__fft_coefficient__coeff_1__attr__abs__y"])))))/2.0))), ((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) * (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))))), ((data["3__kurtosis_x"])))) +
                0.100000*np.tanh((((np.minimum(((((data["flux_ratio_sq_skew"]) / 2.0))), ((data["1__skewness_y"])))) + (data["1__skewness_y"]))/2.0)) +
                0.100000*np.tanh(np.maximum(((data["flux_dif3"])), ((data["detected_flux_ratio_sq_sum"])))) +
                0.100000*np.tanh(np.where(data["flux_max"] > -1, -3.0, data["2__skewness_x"] )) +
                0.100000*np.tanh(((((np.where(((-2.0) - (data["distmod"]))<0, -2.0, data["ddf"] )) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, data["0__skewness_y"], 3.0 )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb3"]>0, ((data["4__kurtosis_x"]) * 2.0), ((np.where(data["flux_d1_pb3"]>0, ((data["detected_flux_median"]) + (data["ddf"])), data["flux_d1_pb3"] )) / 2.0) )))

    def GP_class_67(self,data):
        return (-1.801807 +
                0.100000*np.tanh(((data["4__skewness_x"]) + (data["flux_by_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((((data["4__kurtosis_x"]) + (np.minimum(((data["5__kurtosis_x"])), ((data["4__kurtosis_x"])))))) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((data["3__skewness_x"]) + (((data["5__kurtosis_x"]) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(((((((data["4__kurtosis_x"]) + (data["detected_flux_min"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((((data["distmod"]) + (data["4__skewness_x"]))) + (((data["detected_flux_min"]) + (data["3__kurtosis_x"]))))) + (data["distmod"]))) +
                0.100000*np.tanh(((((data["flux_by_flux_ratio_sq_skew"]) + (data["distmod"]))) + (((data["4__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["4__kurtosis_x"]) + (data["distmod"]))))))) +
                0.100000*np.tanh(((((data["distmod"]) + (data["4__kurtosis_x"]))) + (((((data["3__kurtosis_x"]) + (data["4__kurtosis_x"]))) + (data["distmod"]))))) +
                0.100000*np.tanh(((((((((data["3__kurtosis_x"]) + (data["distmod"]))) + (data["3__kurtosis_x"]))) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["flux_by_flux_ratio_sq_skew"]) + (data["flux_by_flux_ratio_sq_skew"]))) + ((((((data["flux_by_flux_ratio_sq_skew"]) + (((((data["flux_by_flux_ratio_sq_skew"]) + (data["flux_by_flux_ratio_sq_skew"]))) * 2.0)))/2.0)) * 2.0)))) +
                0.100000*np.tanh(((((((((data["5__kurtosis_x"]) - (0.0))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.minimum(((data["distmod"])), ((data["distmod"])))) + (data["3__kurtosis_x"]))) +
                0.100000*np.tanh(np.minimum(((data["3__kurtosis_x"])), ((data["hostgal_photoz_err"])))) +
                0.100000*np.tanh(((((((data["detected_flux_w_mean"]) + (data["hostgal_photoz_err"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh((((((((((data["hostgal_photoz_err"]) + (data["flux_dif2"]))) + (data["4__kurtosis_x"]))/2.0)) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) + (data["distmod"]))) +
                0.100000*np.tanh(((np.tanh((((data["flux_ratio_sq_skew"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))))) - (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["hostgal_photoz_err"])), ((np.minimum(((data["4__fft_coefficient__coeff_1__attr__abs__x"])), ((np.minimum(((np.minimum(((data["flux_dif2"])), ((data["detected_flux_min"]))))), ((data["2__kurtosis_x"]))))))))))), ((data["flux_ratio_sq_skew"])))) +
                0.100000*np.tanh(((data["4__kurtosis_x"]) - (data["0__skewness_x"]))) +
                0.100000*np.tanh(np.minimum(((((data["2__kurtosis_y"]) + (((data["2__skewness_x"]) + (np.minimum(((data["ddf"])), ((data["4__kurtosis_x"]))))))))), ((np.minimum(((data["distmod"])), ((data["distmod"]))))))) +
                0.100000*np.tanh((((((data["distmod"]) + (((((data["distmod"]) * 2.0)) + (((data["mjd_size"]) - (data["detected_mjd_size"]))))))/2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["distmod"])), ((np.minimum(((data["distmod"])), ((data["distmod"])))))))), ((data["5__kurtosis_x"])))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["4__kurtosis_x"])), ((data["5__kurtosis_x"])))) +
                0.100000*np.tanh(((((((((data["5__kurtosis_x"]) + (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["flux_d1_pb0"]))) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((np.minimum(((data["flux_dif2"])), ((data["flux_dif2"])))) + (data["distmod"]))) +
                0.100000*np.tanh(((((data["flux_by_flux_ratio_sq_skew"]) + (data["2__kurtosis_x"]))) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((((((data["flux_dif2"]) + (data["3__skewness_y"]))/2.0)) + (np.minimum(((data["0__kurtosis_y"])), ((data["3__skewness_y"])))))/2.0)) +
                0.100000*np.tanh(((((data["detected_flux_min"]) - (data["detected_mjd_diff"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((np.minimum(((data["distmod"])), ((np.minimum(((((data["2__kurtosis_y"]) + (data["flux_d1_pb5"])))), ((data["2__kurtosis_y"]))))))) / 2.0)) +
                0.100000*np.tanh(np.minimum(((((((data["flux_by_flux_ratio_sq_skew"]) - (data["detected_mjd_diff"]))) * 2.0))), ((data["4__skewness_x"])))) +
                0.100000*np.tanh(((data["5__kurtosis_x"]) - (((np.where(data["detected_mjd_diff"] > -1, ((((data["detected_mjd_diff"]) - (data["3__skewness_x"]))) * 2.0), data["3__skewness_x"] )) * 2.0)))) +
                0.100000*np.tanh((((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (data["distmod"]))/2.0)) + (data["distmod"]))) +
                0.100000*np.tanh(((np.tanh((data["4__skewness_x"]))) - (((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (data["1__skewness_y"]))))) +
                0.100000*np.tanh(np.minimum(((data["4__kurtosis_x"])), ((data["3__kurtosis_x"])))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((np.where(data["detected_mjd_diff"] > -1, data["detected_flux_dif2"], data["detected_flux_dif3"] )) - (data["detected_mjd_diff"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((((np.minimum(((data["hostgal_photoz_err"])), ((data["1__skewness_y"])))) + (data["detected_flux_std"]))) + (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((((data["5__kurtosis_x"]) - (data["detected_mjd_diff"]))) - ((((data["0__kurtosis_x"]) + (data["detected_mjd_diff"]))/2.0)))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((np.minimum(((data["flux_d1_pb5"])), ((np.minimum(((data["mjd_size"])), ((data["2__kurtosis_x"]))))))) + (((data["2__kurtosis_y"]) + (data["distmod"]))))) +
                0.100000*np.tanh(((data["detected_flux_std"]) * ((((14.21325874328613281)) * (((data["flux_dif2"]) * ((7.0)))))))) +
                0.100000*np.tanh(np.where(data["flux_d1_pb5"] > -1, ((data["detected_flux_dif3"]) - (data["0__skewness_x"])), ((np.where(data["3__kurtosis_x"]>0, data["flux_d1_pb5"], data["0__fft_coefficient__coeff_0__attr__abs__y"] )) - (data["flux_d1_pb5"])) )) +
                0.100000*np.tanh(((data["mjd_size"]) + (data["flux_d0_pb5"]))) +
                0.100000*np.tanh(((((((((data["detected_flux_dif2"]) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) + (data["detected_flux_diff"]))) +
                0.100000*np.tanh((((((data["ddf"]) + (data["distmod"]))/2.0)) + (((data["2__kurtosis_y"]) + (((data["detected_flux_dif3"]) + (data["4__kurtosis_y"]))))))) +
                0.100000*np.tanh(((((data["flux_d0_pb1"]) + ((((7.57827568054199219)) - (data["detected_flux_std"]))))) * (((((((data["detected_flux_std"]) - (data["flux_d0_pb1"]))) * 2.0)) * 2.0)))) +
                0.100000*np.tanh(((((data["distmod"]) + (data["flux_dif2"]))) + (data["flux_dif2"]))) +
                0.100000*np.tanh(np.minimum(((-1.0)), ((np.where(data["3__skewness_y"]>0, data["3__skewness_y"], data["3__skewness_y"] ))))) +
                0.100000*np.tanh(((np.maximum(((data["flux_dif2"])), (((14.82015228271484375))))) * ((((((14.82015228271484375)) * ((14.82015228271484375)))) * (data["flux_dif2"]))))) +
                0.100000*np.tanh(((((data["distmod"]) + (data["detected_flux_std"]))) * ((((8.74618148803710938)) + (data["detected_flux_std"]))))) +
                0.100000*np.tanh(np.minimum(((data["3__skewness_y"])), ((((((data["flux_d0_pb5"]) + (data["flux_d0_pb5"]))) * 2.0))))) +
                0.100000*np.tanh(((((((((data["flux_max"]) - (data["detected_mjd_diff"]))) - (data["detected_flux_err_median"]))) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_std"]) * (((data["detected_flux_std"]) * (((data["2__fft_coefficient__coeff_1__attr__abs__y"]) * (data["4__fft_coefficient__coeff_1__attr__abs__x"]))))))) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_1__attr__abs__x"]) + (data["2__skewness_y"]))) +
                0.100000*np.tanh(np.where(((data["detected_flux_dif2"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))>0, ((np.where(data["detected_mean"]>0, data["5__fft_coefficient__coeff_0__attr__abs__y"], data["detected_flux_std"] )) - (data["1__fft_coefficient__coeff_1__attr__abs__x"])), data["1__fft_coefficient__coeff_0__attr__abs__x"] )) +
                0.100000*np.tanh(np.maximum(((((data["detected_flux_dif2"]) + (data["ddf"])))), ((data["detected_flux_std"])))) +
                0.100000*np.tanh(((((((((((data["detected_flux_dif2"]) - (data["detected_mjd_diff"]))) * 2.0)) * 2.0)) - (((data["detected_mjd_diff"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(np.where(np.where(data["flux_dif2"]<0, np.where(data["detected_flux_std"]<0, data["detected_flux_err_skew"], data["flux_max"] ), data["detected_flux_std"] )<0, data["0__kurtosis_x"], data["detected_flux_dif2"] )) +
                0.100000*np.tanh(((((((data["flux_d1_pb5"]) * (data["flux_d0_pb3"]))) - (np.where(data["0__skewness_x"] > -1, data["0__fft_coefficient__coeff_0__attr__abs__y"], data["flux_d1_pb0"] )))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["detected_flux_err_min"]<0, data["5__fft_coefficient__coeff_0__attr__abs__y"], np.minimum(((data["distmod"])), ((data["5__fft_coefficient__coeff_0__attr__abs__y"]))) )) - (data["flux_d1_pb1"]))) +
                0.100000*np.tanh(np.where((((data["flux_max"]) > (data["1__fft_coefficient__coeff_0__attr__abs__x"]))*1.)>0, ((data["distmod"]) + (data["detected_flux_std"])), -2.0 )) +
                0.100000*np.tanh(((((data["detected_flux_dif2"]) - (np.where(data["detected_flux_dif2"] > -1, data["detected_mjd_diff"], (((data["detected_flux_dif2"]) > (data["detected_mjd_diff"]))*1.) )))) - (data["detected_mean"]))) +
                0.100000*np.tanh(((((((data["5__kurtosis_y"]) - (data["flux_by_flux_ratio_sq_sum"]))) + (((((data["1__skewness_y"]) - (data["detected_flux_err_min"]))) + (data["5__kurtosis_y"]))))) - (data["detected_flux_err_median"]))) +
                0.100000*np.tanh(((((((((data["flux_max"]) * 2.0)) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_0__attr__abs__y"] > -1, ((data["5__skewness_x"]) * (data["2__fft_coefficient__coeff_0__attr__abs__y"])), data["2__skewness_x"] )) +
                0.100000*np.tanh(np.where(((data["distmod"]) - (data["flux_d1_pb1"]))<0, -2.0, data["flux_dif2"] )) +
                0.100000*np.tanh(np.where(((data["detected_flux_std"]) - (data["flux_d0_pb2"]))>0, np.where(data["flux_err_std"]>0, data["detected_flux_std"], data["1__kurtosis_x"] ), data["flux_err_std"] )) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb4"]>0, np.where(data["hostgal_photoz"] > -1, np.tanh((data["detected_flux_std"])), data["hostgal_photoz"] ), ((data["hostgal_photoz_err"]) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(np.where(np.where(data["flux_dif2"]<0, data["flux_dif2"], data["flux_err_skew"] )<0, np.where(data["flux_err_skew"]<0, data["flux_err_skew"], data["flux_dif2"] ), data["ddf"] )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb2"]<0, np.where(data["flux_d0_pb4"]<0, data["flux_err_skew"], (((data["flux_d0_pb1"]) < (data["flux_std"]))*1.) ), ((data["detected_flux_std"]) * 2.0) )) +
                0.100000*np.tanh(((((((((((((data["detected_flux_dif2"]) - (data["detected_mjd_diff"]))) * 2.0)) * 2.0)) - (data["detected_mjd_diff"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.maximum(((-3.0)), ((((data["0__kurtosis_y"]) * 2.0))))) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((np.where(data["detected_flux_dif2"]<0, data["0__skewness_x"], ((((data["detected_flux_std"]) * (data["3__skewness_y"]))) - (data["0__skewness_x"])) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["flux_err_max"])), ((((data["0__kurtosis_x"]) + (((np.minimum(((data["flux_err_max"])), ((((data["5__kurtosis_y"]) / 2.0))))) + (data["5__kurtosis_y"])))))))) +
                0.100000*np.tanh(((np.where(np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["flux_d1_pb5"], data["2__fft_coefficient__coeff_1__attr__abs__x"] ) > -1, data["flux_d1_pb5"], data["2__skewness_y"] )) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((np.where(data["flux_d0_pb4"]<0, data["4__fft_coefficient__coeff_0__attr__abs__x"], ((data["detected_flux_std"]) + (np.minimum(((data["distmod"])), ((data["detected_flux_std"]))))) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, (((((np.tanh((data["detected_flux_dif2"]))) > (data["detected_mjd_diff"]))*1.)) - (data["detected_mjd_diff"])), data["detected_mjd_diff"] )) +
                0.100000*np.tanh(np.where(data["distmod"]<0, np.where(data["detected_mjd_diff"]>0, data["flux_err_skew"], data["4__fft_coefficient__coeff_1__attr__abs__x"] ), data["hostgal_photoz_err"] )) +
                0.100000*np.tanh(((np.where(data["2__skewness_y"]>0, data["3__kurtosis_x"], np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), ((data["flux_diff"]))) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, ((((((data["detected_flux_std"]) > (data["flux_d0_pb1"]))*1.)) > (data["flux_d0_pb1"]))*1.), data["flux_d0_pb1"] )) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (data["2__skewness_x"]))) + (data["0__kurtosis_x"]))))) * (data["detected_flux_diff"]))) +
                0.100000*np.tanh(((np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__x"], ((data["detected_flux_std"]) * 2.0) )) - (((data["1__fft_coefficient__coeff_0__attr__abs__x"]) * (data["1__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(np.where(data["hostgal_photoz_err"]<0, data["2__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["hostgal_photoz_err"]<0, data["2__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["3__fft_coefficient__coeff_1__attr__abs__y"]>0, data["4__skewness_x"], data["flux_min"] ) ) )) +
                0.100000*np.tanh(np.where(np.where((((data["distmod"]) > (data["1__fft_coefficient__coeff_1__attr__abs__y"]))*1.)>0, data["flux_dif2"], data["1__fft_coefficient__coeff_1__attr__abs__y"] )>0, (((data["3__fft_coefficient__coeff_1__attr__abs__x"]) > (data["1__fft_coefficient__coeff_1__attr__abs__y"]))*1.), -3.0 )) +
                0.100000*np.tanh(((((((((((data["detected_flux_dif2"]) - (data["detected_mjd_diff"]))) * 2.0)) * 2.0)) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["flux_std"])), ((((data["flux_d0_pb1"]) * (np.minimum(((data["flux_max"])), ((np.tanh((data["detected_flux_diff"]))))))))))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"] > -1, (((((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) < (data["flux_max"]))*1.)) * 2.0)) - (data["1__fft_coefficient__coeff_0__attr__abs__y"])), data["1__fft_coefficient__coeff_0__attr__abs__y"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_d1_pb0"]<0, data["0__fft_coefficient__coeff_0__attr__abs__y"], ((((data["1__skewness_y"]) * (data["detected_flux_w_mean"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__y"])) )) +
                0.100000*np.tanh(np.where(np.where(data["hostgal_photoz_err"]<0, np.where(data["detected_mean"] > -1, data["0__skewness_x"], data["hostgal_photoz_err"] ), data["hostgal_photoz_err"] )<0, data["hostgal_photoz_err"], data["2__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(((data["detected_flux_dif2"]) + (((((data["4__skewness_y"]) * (data["4__skewness_y"]))) + (((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) * (data["4__skewness_y"]))) + (data["distmod"]))))))) +
                0.100000*np.tanh(((((((((data["detected_flux_diff"]) * (data["1__skewness_y"]))) - (data["flux_d1_pb0"]))) - (data["detected_mjd_diff"]))) - (((data["flux_d1_pb0"]) - (data["detected_flux_mean"]))))) +
                0.100000*np.tanh(((((np.where(data["flux_d0_pb0"]>0, ((data["detected_flux_std"]) - (data["flux_d0_pb0"])), ((data["flux_err_skew"]) - (data["1__skewness_x"])) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(data["5__fft_coefficient__coeff_0__attr__abs__y"] > -1, ((data["2__fft_coefficient__coeff_1__attr__abs__y"]) - (data["1__fft_coefficient__coeff_0__attr__abs__x"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * (data["1__fft_coefficient__coeff_1__attr__abs__x"])) )) * (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.where(data["detected_mjd_diff"]<0, data["flux_d0_pb4"], np.where(data["detected_mjd_diff"]>0, np.where(data["flux_d0_pb0"]<0, data["flux_err_min"], data["3__fft_coefficient__coeff_0__attr__abs__x"] ), data["detected_mjd_diff"] ) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]>0, data["4__kurtosis_y"], ((data["flux_d1_pb3"]) + (((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (np.where(data["5__skewness_x"]>0, data["flux_d1_pb3"], data["4__skewness_x"] ))))) )) +
                0.100000*np.tanh(((((np.where(np.where(data["detected_flux_diff"]<0, data["flux_max"], data["flux_diff"] )<0, data["flux_diff"], data["flux_max"] )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"]<0, data["0__fft_coefficient__coeff_1__attr__abs__y"], np.where(data["3__kurtosis_x"]<0, data["detected_flux_err_mean"], ((data["distmod"]) - (data["detected_flux_err_median"])) ) )) +
                0.100000*np.tanh((((((data["detected_mjd_diff"]) < (np.tanh((data["detected_flux_dif2"]))))*1.)) * 2.0)) +
                0.100000*np.tanh(np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"]<0, data["flux_d0_pb5"], ((data["3__fft_coefficient__coeff_1__attr__abs__y"]) - (data["detected_flux_ratio_sq_skew"])) )) +
                0.100000*np.tanh(np.where(np.maximum(((data["2__skewness_x"])), ((data["5__fft_coefficient__coeff_1__attr__abs__y"]))) > -1, np.where(data["mjd_diff"] > -1, data["0__fft_coefficient__coeff_0__attr__abs__x"], data["detected_mjd_size"] ), data["2__skewness_y"] )) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"]<0, (-1.0*((data["4__skewness_y"]))), (((data["3__skewness_x"]) + (data["flux_d1_pb1"]))/2.0) )) +
                0.100000*np.tanh(((data["flux_err_max"]) + (((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (data["3__skewness_y"]))))) +
                0.100000*np.tanh(((np.where(data["detected_flux_by_flux_ratio_sq_skew"]<0, data["detected_flux_err_skew"], ((((np.where(data["flux_std"]<0, data["hostgal_photoz_err"], data["flux_err_max"] )) * 2.0)) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(np.where(((data["distmod"]) + (((data["detected_flux_std"]) + (data["distmod"]))))<0, data["distmod"], ((data["detected_flux_std"]) - (data["distmod"])) )) +
                0.100000*np.tanh(np.where((((data["detected_mjd_diff"]) < (np.tanh(((((data["detected_flux_dif2"]) + (data["detected_mjd_diff"]))/2.0)))))*1.)>0, data["detected_flux_dif2"], -3.0 )) +
                0.100000*np.tanh(np.where(data["2__fft_coefficient__coeff_0__attr__abs__y"]>0, ((data["3__fft_coefficient__coeff_1__attr__abs__y"]) + (data["5__fft_coefficient__coeff_0__attr__abs__y"])), ((data["2__fft_coefficient__coeff_0__attr__abs__y"]) - (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_d1_pb5"])))) )) +
                0.100000*np.tanh(((((((((data["5__kurtosis_y"]) + (((data["flux_d1_pb5"]) + (data["distmod"]))))) + (data["flux_median"]))) + (data["1__kurtosis_y"]))) + (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(data["5__skewness_y"]<0, ((((data["5__kurtosis_y"]) + (data["distmod"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"])), data["1__skewness_y"] )) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"]<0, data["mjd_diff"], ((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"])) )) +
                0.100000*np.tanh(np.where((((data["detected_flux_w_mean"]) < (np.where(data["flux_mean"]<0, data["flux_d1_pb4"], data["flux_mean"] )))*1.)>0, (-1.0*((data["flux_d1_pb4"]))), data["flux_d1_pb4"] )) +
                0.100000*np.tanh(((((data["flux_skew"]) - (data["3__kurtosis_y"]))) - (np.maximum(((data["flux_d1_pb1"])), ((data["3__kurtosis_y"])))))) +
                0.100000*np.tanh(((data["4__skewness_y"]) * (((data["3__fft_coefficient__coeff_1__attr__abs__x"]) - (np.maximum(((data["flux_d0_pb4"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))))))) +
                0.100000*np.tanh(((np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"]<0, np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"]<0, ((data["4__fft_coefficient__coeff_0__attr__abs__y"]) * (data["2__kurtosis_x"])), data["3__kurtosis_x"] ), data["3__kurtosis_x"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_std"]>0, np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"]<0, data["1__kurtosis_x"], data["flux_std"] ), data["detected_flux_skew"] )) +
                0.100000*np.tanh(np.where(data["flux_std"]<0, ((((data["hostgal_photoz_err"]) - (data["detected_flux_by_flux_ratio_sq_skew"]))) - (data["detected_flux_by_flux_ratio_sq_skew"])), np.where(data["detected_flux_by_flux_ratio_sq_skew"]<0, data["detected_flux_by_flux_ratio_sq_skew"], data["4__skewness_y"] ) )) +
                0.100000*np.tanh(((((((((((data["flux_max"]) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(data["detected_flux_err_skew"] > -1, ((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_ratio_sq_skew"]))) * 2.0), 2.0 )) * 2.0)) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"]<0, np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]<0, ((data["distmod"]) - (data["0__fft_coefficient__coeff_0__attr__abs__x"])), np.tanh((data["3__fft_coefficient__coeff_1__attr__abs__x"])) ), data["detected_flux_by_flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(((((((data["detected_flux_dif2"]) - (data["detected_flux_err_median"]))) * 2.0)) + (data["4__kurtosis_y"]))) +
                0.100000*np.tanh(np.where(np.where(((data["flux_err_std"]) + ((((data["1__skewness_y"]) + (data["flux_err_std"]))/2.0)))>0, data["0__fft_coefficient__coeff_1__attr__abs__x"], data["0__fft_coefficient__coeff_1__attr__abs__x"] )>0, data["flux_err_std"], data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(((data["distmod"]) * 2.0) > -1, (((((-1.0*((data["0__fft_coefficient__coeff_1__attr__abs__x"])))) * 2.0)) * 2.0), ((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)) * 2.0) )) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0)) * (np.where(((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) * (data["detected_flux_err_skew"]))) + (data["detected_flux_err_skew"]))>0, data["detected_flux_err_skew"], data["2__fft_coefficient__coeff_1__attr__abs__y"] )))) +
                0.100000*np.tanh(np.where(data["flux_median"]>0, data["detected_flux_diff"], np.where(data["flux_median"]>0, data["3__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["flux_median"] > -1, data["flux_median"], data["flux_median"] ) ) )))

    def GP_class_88(self,data):
        return (-1.503109 +
                0.100000*np.tanh(((((data["distmod"]) + (((((data["distmod"]) * 2.0)) * 2.0)))) + (np.where(data["distmod"] > -1, data["4__fft_coefficient__coeff_0__attr__abs__x"], data["distmod"] )))) +
                0.100000*np.tanh(((np.where(data["4__kurtosis_x"]>0, data["distmod"], data["distmod"] )) - (data["4__kurtosis_x"]))) +
                0.100000*np.tanh(((data["distmod"]) + (((data["distmod"]) + (((data["distmod"]) - (data["3__kurtosis_x"]))))))) +
                0.100000*np.tanh((((((np.minimum(((data["distmod"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) + (((data["distmod"]) + (((data["distmod"]) - (data["2__kurtosis_x"]))))))/2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["distmod"]) + ((((((data["distmod"]) > (data["distmod"]))*1.)) - (data["4__kurtosis_x"]))))) - (data["4__kurtosis_x"]))) +
                0.100000*np.tanh(((((((data["3__fft_coefficient__coeff_1__attr__abs__y"]) + (data["distmod"]))) + (data["distmod"]))) + (((data["distmod"]) * 2.0)))) +
                0.100000*np.tanh(((np.where(((data["distmod"]) - (data["distmod"])) > -1, np.minimum(((data["distmod"])), ((data["distmod"]))), ((data["0__fft_coefficient__coeff_1__attr__abs__y"]) / 2.0) )) - (data["4__kurtosis_x"]))) +
                0.100000*np.tanh(((((((data["distmod"]) + (((np.where(data["2__skewness_x"]>0, -3.0, ((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["distmod"])) )) * 2.0)))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((((np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["distmod"])))) * 2.0))), ((((data["distmod"]) - (data["2__kurtosis_x"])))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["detected_mean"]) - (data["detected_flux_min"]))) +
                0.100000*np.tanh(((((np.where(data["distmod"] > -1, data["detected_mjd_diff"], -1.0 )) + (-1.0))) * 2.0)) +
                0.100000*np.tanh(((((((((((data["flux_skew"]) - (data["flux_skew"]))) - (data["flux_skew"]))) + (data["detected_mjd_diff"]))) - (((data["flux_skew"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh((((data["detected_mjd_diff"]) + (((((((data["distmod"]) + ((-1.0*((data["4__skewness_x"])))))) * 2.0)) * 2.0)))/2.0)) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (((((data["distmod"]) - (data["2__skewness_x"]))) - (data["3__kurtosis_x"]))))) +
                0.100000*np.tanh(((((data["distmod"]) - (data["4__kurtosis_x"]))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz"] > -1, ((data["3__kurtosis_x"]) - (((((((data["3__skewness_x"]) * 2.0)) * 2.0)) * 2.0))), data["hostgal_photoz"] )) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((((((data["distmod"]) * 2.0)) + (data["distmod"])))), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((((((data["distmod"]) - (data["flux_skew"]))) - (data["1__kurtosis_x"]))) - (data["flux_skew"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["2__skewness_x"]<0, np.where(data["2__skewness_x"] > -1, data["distmod"], ((data["distmod"]) + (data["distmod"])) ), -3.0 )) +
                0.100000*np.tanh(np.where(data["3__skewness_x"]<0, np.where(data["distmod"]<0, np.where(-3.0<0, data["distmod"], data["3__skewness_x"] ), data["detected_mjd_diff"] ), -2.0 )) +
                0.100000*np.tanh(((np.where(np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["distmod"]))) > -1, data["distmod"], data["distmod"] )) * 2.0)) +
                0.100000*np.tanh(((((((np.where(data["hostgal_photoz"] > -1, data["detected_mjd_diff"], data["flux_skew"] )) - (data["detected_flux_median"]))) - (data["3__skewness_x"]))) - (data["flux_skew"]))) +
                0.100000*np.tanh((((((((-1.0*((data["flux_skew"])))) * 2.0)) - (np.where(data["flux_ratio_sq_sum"] > -1, data["3__skewness_x"], ((data["flux_d1_pb2"]) * (data["detected_mjd_size"])) )))) * 2.0)) +
                0.100000*np.tanh(((data["distmod"]) - (data["3__skewness_x"]))) +
                0.100000*np.tanh(((((np.where(data["1__kurtosis_x"]>0, -2.0, np.where(data["1__kurtosis_x"]<0, ((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)) * 2.0), data["1__fft_coefficient__coeff_1__attr__abs__y"] ) )) * 2.0)) / 2.0)) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + (((np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((np.minimum(((data["distmod"])), ((data["distmod"]))))))) * 2.0)))) +
                0.100000*np.tanh(np.where(data["flux_skew"]>0, data["hostgal_photoz"], ((data["distmod"]) * 2.0) )) +
                0.100000*np.tanh(((((((data["4__skewness_x"]) * (data["detected_mean"]))) / 2.0)) - ((((12.28340435028076172)) * (data["4__skewness_x"]))))) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) - (data["detected_flux_diff"]))) - (np.maximum(((data["detected_flux_std"])), ((data["1__skewness_x"])))))) - (data["flux_skew"]))) +
                0.100000*np.tanh(((((np.minimum(((((np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["distmod"])))) * 2.0))), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) * 2.0)) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((data["detected_mean"]) - (data["4__skewness_x"]))) * 2.0)) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - ((((data["detected_mjd_diff"]) > (((data["detected_mjd_diff"]) - (0.367879))))*1.)))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((((data["distmod"]) * 2.0))), ((data["distmod"])))) + (((((data["distmod"]) + (data["detected_mjd_diff"]))) + (data["distmod"]))))) +
                0.100000*np.tanh(np.where(data["3__skewness_x"]>0, np.where(data["flux_skew"]>0, -3.0, data["0__fft_coefficient__coeff_0__attr__abs__y"] ), data["1__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(np.minimum(((((((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)) * 2.0)) * 2.0)) * 2.0))), ((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0))))) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, ((data["detected_mjd_diff"]) - (data["distmod"])), ((data["distmod"]) + (data["distmod"])) )) +
                0.100000*np.tanh(((((data["distmod"]) + (data["distmod"]))) + (data["distmod"]))) +
                0.100000*np.tanh(((((np.minimum(((((data["distmod"]) - (data["1__skewness_x"])))), ((np.where(data["detected_mjd_diff"]>0, data["detected_mjd_diff"], data["flux_skew"] ))))) - (data["flux_d1_pb2"]))) * 2.0)) +
                0.100000*np.tanh(((((((((-1.0) + (data["detected_mjd_diff"]))) + (data["distmod"]))) * 2.0)) + (((-1.0) * 2.0)))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((((((data["flux_err_min"]) + (data["distmod"]))) * 2.0)) + (data["detected_mjd_diff"]))) + (data["distmod"]))) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_ratio_sq_skew"])), ((((np.tanh((((np.tanh((((np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["0__fft_coefficient__coeff_1__attr__abs__x"])))) * 2.0)))) * 2.0)))) * 2.0))))) +
                0.100000*np.tanh(((((data["detected_mean"]) + (data["distmod"]))) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((((((data["detected_mean"]) - (data["detected_flux_by_flux_ratio_sq_sum"]))) - (data["flux_skew"]))) * 2.0)) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - ((((data["detected_mjd_diff"]) > (((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))))*1.)))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["distmod"]))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) + (((data["distmod"]) * 2.0)))) + (((data["distmod"]) + (-3.0))))) +
                0.100000*np.tanh(((((((((data["detected_mean"]) - (data["flux_d0_pb3"]))) - (data["1__kurtosis_x"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__kurtosis_x"]))) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + ((((((((data["distmod"]) + (data["distmod"]))) + (np.minimum(((data["hostgal_photoz"])), ((data["distmod"])))))/2.0)) * 2.0)))) +
                0.100000*np.tanh(((((((((-1.0) + (data["detected_mjd_diff"]))) * 2.0)) * 2.0)) + (-1.0))) +
                0.100000*np.tanh(np.where(np.minimum(((data["detected_mjd_diff"])), ((np.minimum(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_flux_ratio_sq_skew"])))), ((data["detected_flux_ratio_sq_skew"]))))))>0, data["detected_mjd_diff"], data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - ((((np.where(data["flux_skew"] > -1, data["detected_mjd_diff"], data["flux_err_min"] )) > (((data["detected_mjd_diff"]) - (data["4__fft_coefficient__coeff_0__attr__abs__y"]))))*1.)))) +
                0.100000*np.tanh(((data["flux_err_min"]) + (((((data["detected_mjd_diff"]) + (((data["flux_err_min"]) + (data["detected_mjd_diff"]))))) + (data["flux_err_min"]))))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_err_skew"]))) +
                0.100000*np.tanh(((data["distmod"]) - (data["1__kurtosis_x"]))) +
                0.100000*np.tanh(((((((np.minimum(((np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__x"])), ((data["3__fft_coefficient__coeff_1__attr__abs__x"]))))), ((data["detected_mjd_diff"])))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["distmod"]) + (((data["0__skewness_x"]) - ((((data["0__skewness_x"]) > (data["detected_mjd_diff"]))*1.)))))) + (data["distmod"]))) +
                0.100000*np.tanh(((np.where(data["flux_skew"]>0, -2.0, ((((data["flux_err_min"]) * 2.0)) * 2.0) )) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, data["detected_mjd_diff"], ((data["distmod"]) + (np.where(data["distmod"] > -1, data["distmod"], np.minimum(((data["flux_by_flux_ratio_sq_sum"])), ((data["flux_d0_pb5"]))) ))) )) +
                0.100000*np.tanh(((((np.where(((data["flux_err_median"]) * 2.0) > -1, ((data["detected_mean"]) - (data["3__fft_coefficient__coeff_0__attr__abs__y"])), np.minimum(((data["detected_mean"])), ((data["detected_mean"]))) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((data["distmod"]) + ((((data["detected_mjd_diff"]) + (((((data["flux_d0_pb0"]) + (data["detected_mjd_diff"]))) + (data["distmod"]))))/2.0)))) + (data["distmod"]))) +
                0.100000*np.tanh(np.where((((data["2__fft_coefficient__coeff_0__attr__abs__x"]) + (data["detected_flux_by_flux_ratio_sq_sum"]))/2.0) > -1, ((data["detected_mjd_diff"]) - (data["4__fft_coefficient__coeff_1__attr__abs__x"])), ((data["2__fft_coefficient__coeff_1__attr__abs__y"]) - (data["detected_mjd_diff"])) )) +
                0.100000*np.tanh(((((data["detected_flux_dif3"]) + (((((((data["flux_err_min"]) + (data["detected_mean"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)))) + (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["flux_err_min"], (-1.0*((data["1__fft_coefficient__coeff_1__attr__abs__y"]))) )) + (data["detected_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_ratio_sq_skew"] > -1, ((((((((data["flux_median"]) - (data["3__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["3__fft_coefficient__coeff_0__attr__abs__y"]))) + (data["flux_d0_pb0"]))) * 2.0), data["flux_d0_pb2"] )) +
                0.100000*np.tanh(((np.minimum(((data["detected_flux_ratio_sq_skew"])), ((((((((data["distmod"]) + (data["distmod"]))) + (data["detected_flux_ratio_sq_skew"]))) + (data["distmod"])))))) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.where(data["flux_median"]<0, data["0__fft_coefficient__coeff_1__attr__abs__x"], ((data["detected_mjd_diff"]) + (np.tanh((((data["flux_err_min"]) * 2.0))))) )) +
                0.100000*np.tanh(np.where(np.where(data["detected_flux_err_skew"]<0, data["detected_flux_diff"], data["detected_mjd_diff"] ) > -1, data["detected_mjd_diff"], data["detected_flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(np.minimum(((((((np.minimum(((data["detected_mjd_diff"])), ((data["0__fft_coefficient__coeff_1__attr__abs__x"])))) * 2.0)) * 2.0))), ((((data["detected_mean"]) * 2.0))))) +
                0.100000*np.tanh(np.minimum(((((((((data["flux_median"]) - (data["detected_flux_max"]))) - (data["flux_std"]))) - (data["1__kurtosis_x"])))), ((data["flux_median"])))) +
                0.100000*np.tanh(((data["flux_d0_pb5"]) + (((np.where(data["detected_mjd_diff"] > -1, ((-2.0) + (data["detected_mjd_diff"])), data["detected_mjd_diff"] )) * 2.0)))) +
                0.100000*np.tanh(((((data["flux_err_min"]) + (((data["hostgal_photoz"]) + (((data["flux_err_min"]) + (data["distmod"]))))))) * 2.0)) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["distmod"]))) +
                0.100000*np.tanh((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (((np.where(data["detected_mean"]<0, ((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__y"])), data["0__fft_coefficient__coeff_0__attr__abs__x"] )) * 2.0)))/2.0)) +
                0.100000*np.tanh(((np.where(data["flux_err_max"] > -1, data["0__fft_coefficient__coeff_1__attr__abs__x"], data["distmod"] )) + (np.minimum(((data["hostgal_photoz"])), ((data["0__fft_coefficient__coeff_0__attr__abs__y"])))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["flux_err_min"])), ((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) * (data["distmod"]))))))), ((data["distmod"])))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_1__attr__abs__y"]) * (((data["detected_mjd_diff"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_1__attr__abs__y"]) - (data["flux_err_min"]))) +
                0.100000*np.tanh(((((np.where(data["0__skewness_y"]>0, data["detected_flux_ratio_sq_skew"], np.where(data["flux_err_min"] > -1, data["flux_err_min"], data["detected_flux_ratio_sq_skew"] ) )) * 2.0)) + (data["distmod"]))) +
                0.100000*np.tanh(((((data["distmod"]) - (data["flux_dif3"]))) + (((data["4__skewness_y"]) + (((((data["detected_mjd_diff"]) - (data["detected_flux_mean"]))) / 2.0)))))) +
                0.100000*np.tanh(np.where(np.maximum(((data["flux_d1_pb1"])), ((((((data["detected_mjd_diff"]) * (data["detected_mjd_diff"]))) - (data["detected_flux_diff"]))))) > -1, data["detected_flux_ratio_sq_skew"], data["5__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(data["flux_d0_pb1"]<0, data["2__fft_coefficient__coeff_1__attr__abs__x"], np.where((((-1.0*((data["0__fft_coefficient__coeff_0__attr__abs__x"])))) / 2.0) > -1, data["0__fft_coefficient__coeff_0__attr__abs__y"], data["flux_d0_pb1"] ) )) +
                0.100000*np.tanh((((data["distmod"]) < (np.minimum(((data["distmod"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))))*1.)) +
                0.100000*np.tanh((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (((np.minimum(((data["flux_err_min"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) + (np.minimum(((data["4__skewness_y"])), ((data["detected_mean"])))))))/2.0)) +
                0.100000*np.tanh(((((data["flux_err_min"]) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["distmod"]) + ((((data["distmod"]) + ((((((data["distmod"]) + (data["distmod"]))) + (data["flux_d1_pb4"]))/2.0)))/2.0)))) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - (data["2__skewness_y"]))) +
                0.100000*np.tanh(((((((data["detected_mean"]) + (data["distmod"]))) * 2.0)) - (np.where(data["mjd_diff"] > -1, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["5__fft_coefficient__coeff_1__attr__abs__y"] )))) +
                0.100000*np.tanh(np.where(data["flux_d1_pb4"] > -1, data["flux_err_min"], ((data["1__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0) )) +
                0.100000*np.tanh(((((data["flux_err_min"]) * 2.0)) / 2.0)) +
                0.100000*np.tanh((((data["distmod"]) + (data["flux_d0_pb0"]))/2.0)) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["detected_flux_err_mean"]) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))))) +
                0.100000*np.tanh((((((data["distmod"]) + (data["distmod"]))/2.0)) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh((((((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)) + (data["detected_flux_diff"]))/2.0)) + (np.maximum(((data["0__fft_coefficient__coeff_1__attr__abs__x"])), ((((data["flux_err_min"]) * 2.0))))))) +
                0.100000*np.tanh(np.tanh((((data["flux_min"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(np.minimum(((((np.minimum(((data["3__skewness_y"])), ((data["3__skewness_y"])))) + (0.367879)))), ((np.minimum(((data["flux_err_min"])), ((((data["5__skewness_y"]) + (data["flux_err_min"]))))))))) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - (np.where(data["flux_d0_pb3"]<0, ((data["5__fft_coefficient__coeff_0__attr__abs__x"]) * (data["3__fft_coefficient__coeff_0__attr__abs__x"])), ((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (data["4__skewness_x"])) )))) +
                0.100000*np.tanh(((data["5__kurtosis_y"]) + (np.minimum(((data["detected_flux_ratio_sq_skew"])), ((((np.tanh((data["4__kurtosis_y"]))) + (data["detected_mjd_diff"])))))))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["5__kurtosis_y"]))) +
                0.100000*np.tanh((((((data["distmod"]) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(data["flux_err_min"] > -1, data["detected_flux_ratio_sq_skew"], ((data["detected_flux_ratio_sq_skew"]) * 2.0) )) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_d0_pb0"]))) + ((((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_median"]))) < (np.tanh((data["0__fft_coefficient__coeff_0__attr__abs__y"]))))*1.)))) +
                0.100000*np.tanh(np.where(((data["distmod"]) + (data["1__skewness_y"]))>0, (((data["detected_flux_ratio_sq_skew"]) > (np.minimum(((data["distmod"])), ((data["5__skewness_y"])))))*1.), data["distmod"] )) +
                0.100000*np.tanh(((((data["3__skewness_y"]) + (data["4__kurtosis_y"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, ((data["flux_err_min"]) - (data["detected_flux_min"])), data["hostgal_photoz"] )) +
                0.100000*np.tanh((((np.where(data["2__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_err_median"], data["0__fft_coefficient__coeff_1__attr__abs__y"] )) + (((data["2__fft_coefficient__coeff_1__attr__abs__y"]) / 2.0)))/2.0)) +
                0.100000*np.tanh(np.where(data["mjd_size"]<0, data["flux_median"], np.where(data["detected_flux_min"]>0, data["0__fft_coefficient__coeff_1__attr__abs__y"], ((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_d1_pb5"]))) + (data["5__kurtosis_x"])) ) )) +
                0.100000*np.tanh(np.maximum(((data["detected_flux_ratio_sq_skew"])), ((data["0__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh(((np.where(data["flux_err_std"] > -1, data["flux_median"], data["0__fft_coefficient__coeff_0__attr__abs__y"] )) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((data["distmod"]) + (np.where(np.tanh((data["detected_flux_min"]))>0, data["detected_flux_err_skew"], data["4__skewness_y"] )))) +
                0.100000*np.tanh(((data["distmod"]) + (((data["distmod"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(np.where(3.0 > -1, data["distmod"], ((data["detected_mean"]) - ((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) < (data["5__fft_coefficient__coeff_1__attr__abs__y"]))*1.))) )) +
                0.100000*np.tanh((((((((((data["3__fft_coefficient__coeff_1__attr__abs__y"]) + (data["3__skewness_y"]))/2.0)) - (data["1__kurtosis_x"]))) - ((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (data["detected_flux_min"]))/2.0)))) - (data["1__kurtosis_x"]))) +
                0.100000*np.tanh(np.where(np.where(data["flux_err_min"]>0, data["flux_err_min"], data["flux_err_min"] )>0, data["4__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["flux_err_min"], data["flux_err_min"] ) )) +
                0.100000*np.tanh(np.minimum(((((data["4__skewness_y"]) + (data["3__skewness_y"])))), ((data["1__fft_coefficient__coeff_1__attr__abs__x"])))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_median"]))) + (data["flux_median"]))) +
                0.100000*np.tanh(np.minimum(((np.where(data["detected_mean"]<0, data["1__fft_coefficient__coeff_1__attr__abs__x"], np.where(data["5__kurtosis_x"]>0, data["detected_mjd_diff"], np.tanh((data["flux_err_min"])) ) ))), ((0.0)))) +
                0.100000*np.tanh(((((((((data["detected_mjd_diff"]) * (data["detected_mjd_diff"]))) - (((data["detected_flux_max"]) * 2.0)))) - (data["detected_flux_min"]))) - (data["flux_d0_pb3"]))))

    def GP_class_90(self,data):
        return (-0.436273 +
                0.100000*np.tanh(np.minimum(((-3.0)), ((np.minimum(((np.minimum(((np.minimum(((-3.0)), ((-3.0))))), ((3.141593))))), ((-3.0))))))) +
                0.100000*np.tanh(((np.minimum(((-3.0)), ((-2.0)))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((-3.0)), ((-3.0)))) +
                0.100000*np.tanh(np.minimum(((data["2__fft_coefficient__coeff_1__attr__abs__y"])), ((np.minimum(((np.minimum(((np.minimum(((-3.0)), ((data["3__kurtosis_y"]))))), ((-3.0))))), ((-3.0))))))) +
                0.100000*np.tanh((((-3.0) + (np.minimum(((np.minimum(((-3.0)), ((np.minimum(((-3.0)), ((data["4__kurtosis_x"])))))))), ((np.minimum(((data["2__skewness_x"])), ((-3.0))))))))/2.0)) +
                0.100000*np.tanh(np.minimum(((np.where(np.minimum(((data["distmod"])), ((data["distmod"])))>0, data["flux_by_flux_ratio_sq_skew"], -2.0 ))), ((data["distmod"])))) +
                0.100000*np.tanh(((((np.minimum(((np.minimum(((((((data["distmod"]) * 2.0)) * 2.0))), ((data["3__kurtosis_x"]))))), ((data["distmod"])))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((((data["flux_ratio_sq_skew"]) + (data["distmod"])))), ((((np.minimum(((data["distmod"])), ((data["distmod"])))) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((np.minimum(((np.minimum(((data["4__kurtosis_x"])), ((data["distmod"]))))), ((data["distmod"])))) + (data["2__skewness_x"])))), ((np.minimum(((data["distmod"])), ((data["4__kurtosis_x"]))))))) +
                0.100000*np.tanh(np.minimum(((-2.0)), ((-2.0)))) +
                0.100000*np.tanh(((((data["4__kurtosis_x"]) + (np.minimum(((data["distmod"])), ((data["4__kurtosis_x"])))))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((-3.0) / 2.0))), ((np.minimum(((data["4__kurtosis_x"])), ((data["3__skewness_x"]))))))) +
                0.100000*np.tanh(((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((((((((data["flux_by_flux_ratio_sq_skew"]) + (data["distmod"]))) * 2.0)) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((((data["distmod"]) * 2.0)))))), ((data["3__kurtosis_x"])))) * 2.0)) +
                0.100000*np.tanh(((data["flux_ratio_sq_skew"]) + (data["distmod"]))) +
                0.100000*np.tanh(((((np.minimum(((data["flux_dif2"])), ((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((((data["flux_min"]) - (data["hostgal_photoz"]))))))))) - (data["detected_flux_err_max"]))) * 2.0)) +
                0.100000*np.tanh(((((data["distmod"]) + (data["distmod"]))) + (((data["distmod"]) + (data["flux_min"]))))) +
                0.100000*np.tanh(np.minimum(((((data["distmod"]) * 2.0))), ((np.minimum(((((((np.minimum(((data["distmod"])), ((data["flux_by_flux_ratio_sq_skew"])))) * 2.0)) * 2.0))), ((data["distmod"]))))))) +
                0.100000*np.tanh(((((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((data["distmod"])))) + (data["flux_min"]))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((((((data["distmod"]) + (data["flux_by_flux_ratio_sq_skew"]))) * 2.0)))))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["flux_by_flux_ratio_sq_skew"]) + (np.minimum(((data["2__kurtosis_x"])), ((data["distmod"])))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["5__skewness_y"])), ((np.minimum(((data["flux_d0_pb3"])), ((data["detected_flux_min"])))))))), (((-1.0*((np.minimum(((data["flux_std"])), ((data["3__kurtosis_x"])))))))))) +
                0.100000*np.tanh(((((((data["hostgal_photoz"]) + (((data["4__kurtosis_x"]) + (data["distmod"]))))) * 2.0)) + (((((data["detected_flux_max"]) / 2.0)) + (data["distmod"]))))) +
                0.100000*np.tanh(((np.minimum(((data["3__kurtosis_x"])), ((((data["distmod"]) + (((((data["3__skewness_x"]) + (data["2__kurtosis_x"]))) - (data["detected_flux_err_std"])))))))) * 2.0)) +
                0.100000*np.tanh(((((((data["3__kurtosis_x"]) - (data["detected_flux_err_max"]))) - (0.367879))) - (data["flux_d0_pb3"]))) +
                0.100000*np.tanh(((np.minimum(((((data["flux_d1_pb2"]) + (data["4__kurtosis_x"])))), ((((data["flux_by_flux_ratio_sq_skew"]) + (data["distmod"])))))) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["4__kurtosis_x"]) + ((((((((data["4__kurtosis_x"]) * 2.0)) + (data["distmod"]))/2.0)) + (((data["distmod"]) / 2.0)))))) + (data["2__skewness_x"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["3__kurtosis_x"])), ((data["flux_d0_pb3"]))))), ((np.minimum(((data["3__kurtosis_x"])), ((data["flux_median"]))))))) +
                0.100000*np.tanh(((((((data["detected_flux_min"]) - (data["detected_flux_err_median"]))) * 2.0)) + (((data["4__kurtosis_x"]) + (data["mjd_diff"]))))) +
                0.100000*np.tanh(np.minimum(((((((data["distmod"]) - (data["hostgal_photoz"]))) + (((data["distmod"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"])))))), ((((data["flux_by_flux_ratio_sq_skew"]) + (data["distmod"])))))) +
                0.100000*np.tanh((((((((data["flux_min"]) + (((data["flux_min"]) * 2.0)))/2.0)) * 2.0)) + (data["flux_d1_pb5"]))) +
                0.100000*np.tanh(((((((((data["distmod"]) * 2.0)) + (data["flux_d0_pb2"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["flux_d0_pb2"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) - ((((((data["flux_d1_pb2"]) - (data["0__skewness_y"]))) < (data["flux_dif2"]))*1.)))) +
                0.100000*np.tanh((((((((((6.0)) / 2.0)) + (data["flux_min"]))) - (data["4__kurtosis_x"]))) - (data["hostgal_photoz"]))) +
                0.100000*np.tanh((((((-1.0*((data["2__fft_coefficient__coeff_1__attr__abs__x"])))) + (((((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) + (data["flux_by_flux_ratio_sq_skew"]))) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["distmod"])), ((np.minimum(((data["distmod"])), ((data["flux_dif2"]))))))) +
                0.100000*np.tanh(((data["4__kurtosis_x"]) + (data["flux_d0_pb1"]))) +
                0.100000*np.tanh(np.minimum(((((data["distmod"]) / 2.0))), ((np.where(data["distmod"]<0, (((data["distmod"]) + ((-1.0*((data["flux_dif2"])))))/2.0), data["1__kurtosis_x"] ))))) +
                0.100000*np.tanh(((data["flux_d0_pb2"]) + (data["flux_d0_pb2"]))) +
                0.100000*np.tanh(np.where(((data["5__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0) > -1, ((data["2__fft_coefficient__coeff_0__attr__abs__y"]) - (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) - (((data["flux_by_flux_ratio_sq_skew"]) - (data["hostgal_photoz"])))))), data["flux_median"] )) +
                0.100000*np.tanh(((data["flux_d0_pb2"]) + (((data["flux_d0_pb2"]) + (((data["distmod"]) * 2.0)))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["flux_dif2"])), ((((data["distmod"]) + (data["distmod"]))))))), ((data["distmod"])))) +
                0.100000*np.tanh((((((data["4__skewness_x"]) + (data["flux_min"]))) + (((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_d0_pb2"]))))/2.0)) +
                0.100000*np.tanh((((data["detected_flux_min"]) + (((data["2__fft_coefficient__coeff_0__attr__abs__y"]) * (data["flux_d1_pb1"]))))/2.0)) +
                0.100000*np.tanh(((((((((data["3__kurtosis_x"]) - (data["detected_mjd_diff"]))) - (data["hostgal_photoz"]))) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((np.tanh((((((((data["distmod"]) * 2.0)) + (data["flux_d0_pb1"]))) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["5__kurtosis_x"])), ((np.minimum(((data["1__skewness_x"])), ((((data["5__kurtosis_x"]) * 2.0)))))))) +
                0.100000*np.tanh(((data["1__skewness_y"]) + (data["distmod"]))) +
                0.100000*np.tanh((((((data["5__kurtosis_x"]) > ((((data["hostgal_photoz"]) < (data["hostgal_photoz"]))*1.)))*1.)) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.where(data["2__kurtosis_x"]>0, data["flux_d0_pb2"], data["2__fft_coefficient__coeff_0__attr__abs__x"] )) * (data["2__kurtosis_x"]))) +
                0.100000*np.tanh(((((data["distmod"]) - (data["hostgal_photoz"]))) - (data["hostgal_photoz"]))) +
                0.100000*np.tanh(np.minimum(((np.where(np.minimum(((data["flux_d0_pb2"])), ((data["detected_flux_min"]))) > -1, data["5__kurtosis_x"], data["flux_ratio_sq_skew"] ))), ((data["flux_d0_pb2"])))) +
                0.100000*np.tanh(((np.minimum(((data["4__fft_coefficient__coeff_0__attr__abs__y"])), ((data["2__fft_coefficient__coeff_0__attr__abs__x"])))) * (data["2__kurtosis_x"]))) +
                0.100000*np.tanh((((data["flux_d0_pb3"]) + (data["flux_d0_pb3"]))/2.0)) +
                0.100000*np.tanh(np.where(data["distmod"]>0, np.where(data["distmod"] > -1, 2.0, data["distmod"] ), -2.0 )) +
                0.100000*np.tanh(((((((-1.0*((data["4__skewness_x"])))) < (np.tanh((data["hostgal_photoz"]))))*1.)) + ((-1.0*((data["hostgal_photoz"])))))) +
                0.100000*np.tanh((((data["flux_d0_pb2"]) + ((((np.tanh((data["1__skewness_y"]))) + (data["1__skewness_y"]))/2.0)))/2.0)) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, np.where(2.0<0, data["flux_err_median"], data["detected_mean"] ), ((data["2__kurtosis_y"]) + (data["4__fft_coefficient__coeff_1__attr__abs__y"])) )) +
                0.100000*np.tanh((((data["flux_by_flux_ratio_sq_skew"]) + (np.where(data["distmod"]>0, data["4__kurtosis_x"], data["distmod"] )))/2.0)) +
                0.100000*np.tanh(((np.where(((data["flux_median"]) + (data["flux_median"]))>0, data["4__fft_coefficient__coeff_1__attr__abs__y"], data["0__fft_coefficient__coeff_0__attr__abs__x"] )) * (data["3__kurtosis_x"]))) +
                0.100000*np.tanh(((data["detected_flux_w_mean"]) * (np.where(data["detected_flux_err_std"]>0, data["ddf"], data["flux_d0_pb4"] )))) +
                0.100000*np.tanh((((((data["distmod"]) + (data["detected_flux_min"]))) > (data["flux_d1_pb5"]))*1.)) +
                0.100000*np.tanh(np.where(data["distmod"]<0, np.where(data["flux_median"] > -1, data["flux_d0_pb1"], data["flux_w_mean"] ), ((((data["distmod"]) - (data["hostgal_photoz"]))) - (data["hostgal_photoz"])) )) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_median"], ((np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, np.maximum(((data["detected_flux_by_flux_ratio_sq_skew"])), ((data["4__fft_coefficient__coeff_1__attr__abs__y"]))), data["detected_flux_w_mean"] )) + (data["detected_flux_by_flux_ratio_sq_sum"])) )) +
                0.100000*np.tanh(np.where(((data["detected_flux_err_median"]) * (data["3__fft_coefficient__coeff_0__attr__abs__x"]))<0, (((data["flux_d0_pb3"]) + (data["flux_diff"]))/2.0), (((data["flux_d0_pb3"]) < (data["1__skewness_y"]))*1.) )) +
                0.100000*np.tanh(((np.where(data["flux_median"]>0, data["flux_by_flux_ratio_sq_skew"], ((data["distmod"]) + (((data["distmod"]) + (data["detected_flux_w_mean"])))) )) * 2.0)) +
                0.100000*np.tanh(((((data["flux_d0_pb2"]) + (((data["flux_d0_pb2"]) * 2.0)))) + (((data["1__skewness_y"]) + (data["detected_flux_err_min"]))))) +
                0.100000*np.tanh(np.minimum(((((data["5__kurtosis_x"]) / 2.0))), ((((np.tanh((data["flux_by_flux_ratio_sq_skew"]))) * (np.tanh((data["flux_median"])))))))) +
                0.100000*np.tanh(np.where(data["mjd_size"] > -1, (((data["flux_d0_pb5"]) < ((((((data["0__kurtosis_y"]) / 2.0)) > (data["flux_d0_pb5"]))*1.)))*1.), data["flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(((((((((((data["detected_flux_max"]) + (data["distmod"]))) + (data["distmod"]))) * 2.0)) + (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh((((data["detected_flux_std"]) + (((data["2__skewness_y"]) + (np.minimum(((data["3__kurtosis_y"])), ((data["flux_d0_pb1"])))))))/2.0)) +
                0.100000*np.tanh((((data["4__fft_coefficient__coeff_1__attr__abs__y"]) > (data["flux_w_mean"]))*1.)) +
                0.100000*np.tanh(((data["flux_dif2"]) * (((data["detected_flux_median"]) - (((data["detected_flux_median"]) * (((data["4__skewness_x"]) + (data["flux_by_flux_ratio_sq_skew"]))))))))) +
                0.100000*np.tanh(((((data["4__kurtosis_x"]) * (np.where(data["0__skewness_x"]<0, data["detected_flux_by_flux_ratio_sq_skew"], data["flux_d0_pb4"] )))) * (((data["flux_d1_pb1"]) * 2.0)))) +
                0.100000*np.tanh(np.where((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) > (data["5__fft_coefficient__coeff_1__attr__abs__x"]))*1.)<0, data["5__fft_coefficient__coeff_1__attr__abs__x"], ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * (data["3__kurtosis_x"])) )) +
                0.100000*np.tanh(((np.where((((data["flux_d0_pb3"]) > (data["flux_d0_pb5"]))*1.)>0, (((data["flux_d0_pb3"]) > (data["flux_dif3"]))*1.), ((data["flux_dif3"]) - (data["flux_d0_pb5"])) )) * 2.0)) +
                0.100000*np.tanh(np.maximum(((((data["detected_flux_by_flux_ratio_sq_skew"]) + (data["detected_flux_by_flux_ratio_sq_skew"])))), ((data["flux_d0_pb3"])))) +
                0.100000*np.tanh(np.where(data["flux_median"]>0, data["flux_median"], data["detected_flux_err_min"] )) +
                0.100000*np.tanh(((data["mjd_size"]) + (np.maximum(((data["detected_flux_err_mean"])), ((data["4__fft_coefficient__coeff_1__attr__abs__y"])))))) +
                0.100000*np.tanh(((((((((data["distmod"]) + (((((data["hostgal_photoz"]) + (data["detected_flux_max"]))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(np.maximum(((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) / 2.0))), ((data["flux_err_skew"]))) > -1, data["5__kurtosis_x"], data["detected_flux_err_median"] )) +
                0.100000*np.tanh(((np.where(data["3__kurtosis_x"] > -1, data["3__kurtosis_x"], ((((data["detected_mjd_size"]) + (data["3__kurtosis_x"]))) * 2.0) )) * (((data["detected_mjd_size"]) * 2.0)))) +
                0.100000*np.tanh(((data["detected_mjd_size"]) * (((((((((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) * 2.0)) * 2.0)) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(((np.tanh((np.where(data["flux_median"] > -1, data["flux_median"], data["flux_median"] )))) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_min"]<0, np.minimum(((data["3__kurtosis_x"])), ((data["flux_d0_pb0"]))), np.where(data["detected_flux_min"]>0, data["4__fft_coefficient__coeff_0__attr__abs__x"], data["detected_flux_dif3"] ) )) +
                0.100000*np.tanh(((((data["1__fft_coefficient__coeff_0__attr__abs__y"]) * (np.where(data["mwebv"]<0, data["3__kurtosis_x"], data["3__fft_coefficient__coeff_1__attr__abs__y"] )))) / 2.0)) +
                0.100000*np.tanh(np.where((((data["distmod"]) + (data["flux_err_skew"]))/2.0)>0, data["1__skewness_y"], data["hostgal_photoz_err"] )) +
                0.100000*np.tanh(((((((((((((data["detected_flux_max"]) + (data["hostgal_photoz"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((np.where(data["2__fft_coefficient__coeff_0__attr__abs__y"]<0, data["5__fft_coefficient__coeff_1__attr__abs__x"], ((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"])) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((((data["0__skewness_x"]) > (data["hostgal_photoz"]))*1.)) > (data["flux_d1_pb0"]))*1.)) +
                0.100000*np.tanh(((((((data["flux_d0_pb0"]) * 2.0)) * (((data["3__kurtosis_x"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_median"]) * (data["flux_skew"]))) +
                0.100000*np.tanh(np.where(data["flux_median"]>0, np.minimum(((data["5__kurtosis_x"])), ((data["1__fft_coefficient__coeff_0__attr__abs__x"]))), np.minimum(((data["flux_d0_pb2"])), ((np.minimum(((data["flux_d1_pb2"])), ((data["flux_d0_pb2"])))))) )) +
                0.100000*np.tanh(np.where(data["flux_by_flux_ratio_sq_skew"] > -1, ((data["flux_by_flux_ratio_sq_skew"]) * (data["detected_flux_ratio_sq_skew"])), np.where(data["flux_ratio_sq_sum"] > -1, ((data["flux_by_flux_ratio_sq_skew"]) * (data["flux_ratio_sq_skew"])), data["0__skewness_x"] ) )) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"] > -1, ((data["detected_flux_err_mean"]) * (data["5__fft_coefficient__coeff_0__attr__abs__x"])), ((data["5__fft_coefficient__coeff_0__attr__abs__x"]) * (data["5__fft_coefficient__coeff_0__attr__abs__x"])) )) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))))) - (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__skewness_x"]))))) +
                0.100000*np.tanh((((data["flux_median"]) > (data["detected_flux_mean"]))*1.)) +
                0.100000*np.tanh(((((((data["flux_d0_pb2"]) + (((data["flux_median"]) + (data["flux_median"]))))) + (((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (data["flux_median"]))))) + (data["flux_median"]))) +
                0.100000*np.tanh(np.where(data["mjd_diff"] > -1, np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]>0, data["4__skewness_x"], data["detected_flux_min"] ), data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(data["flux_min"]<0, np.where(data["detected_flux_err_min"]>0, data["flux_max"], data["mjd_size"] ), data["mwebv"] )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz_err"]<0, ((((data["distmod"]) * 2.0)) * 2.0), data["5__skewness_x"] )) +
                0.100000*np.tanh(((np.where(((data["3__skewness_x"]) * (data["detected_flux_by_flux_ratio_sq_skew"])) > -1, data["flux_err_skew"], data["detected_flux_by_flux_ratio_sq_skew"] )) + (((data["detected_flux_by_flux_ratio_sq_skew"]) * (data["3__kurtosis_x"]))))) +
                0.100000*np.tanh(np.where(data["hostgal_photoz_err"]<0, np.maximum(((data["distmod"])), ((data["3__kurtosis_y"]))), np.tanh((data["flux_d0_pb2"])) )) +
                0.100000*np.tanh((((((((data["flux_d0_pb4"]) / 2.0)) + (data["2__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)) * (data["flux_max"]))) +
                0.100000*np.tanh(np.where(np.where(data["flux_by_flux_ratio_sq_skew"]>0, data["flux_d1_pb1"], data["2__skewness_y"] )>0, data["flux_by_flux_ratio_sq_skew"], (((((data["flux_err_max"]) + (((data["2__kurtosis_y"]) / 2.0)))/2.0)) * 2.0) )) +
                0.100000*np.tanh(np.maximum(((data["hostgal_photoz_err"])), ((np.maximum(((data["hostgal_photoz_err"])), ((np.maximum(((((data["hostgal_photoz_err"]) * (data["flux_diff"])))), (((((data["flux_diff"]) + (data["flux_diff"]))/2.0))))))))))) +
                0.100000*np.tanh(np.where(data["flux_min"] > -1, np.where(data["flux_d0_pb0"]>0, data["flux_skew"], data["flux_d0_pb0"] ), data["detected_flux_by_flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(np.where(data["flux_by_flux_ratio_sq_skew"] > -1, np.where((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) < (data["5__fft_coefficient__coeff_1__attr__abs__x"]))*1.)>0, data["mjd_diff"], data["mjd_size"] ), data["4__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(((((np.where(data["2__fft_coefficient__coeff_1__attr__abs__y"]>0, data["flux_median"], data["detected_flux_err_std"] )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["flux_d0_pb2"]) + (((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))))) +
                0.100000*np.tanh((((np.where(data["detected_mjd_diff"] > -1, data["detected_flux_dif2"], data["detected_flux_dif2"] )) > (data["detected_mjd_diff"]))*1.)) +
                0.100000*np.tanh(np.where(((data["detected_flux_by_flux_ratio_sq_sum"]) - (data["5__fft_coefficient__coeff_1__attr__abs__y"])) > -1, np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["5__fft_coefficient__coeff_1__attr__abs__y"], -2.0 ), -2.0 )) +
                0.100000*np.tanh(np.where(((data["flux_diff"]) + (((data["distmod"]) + (data["distmod"]))))<0, ((data["distmod"]) + (data["hostgal_photoz_err"])), 2.718282 )) +
                0.100000*np.tanh(((data["4__skewness_x"]) * (((np.where(data["3__kurtosis_x"]>0, data["3__kurtosis_x"], ((data["3__kurtosis_x"]) * (data["flux_ratio_sq_sum"])) )) * (data["flux_ratio_sq_sum"]))))) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_0__attr__abs__x"]<0, np.where(data["flux_err_mean"]>0, data["detected_flux_min"], data["flux_ratio_sq_sum"] ), data["2__kurtosis_y"] )) +
                0.100000*np.tanh((((((((data["flux_d0_pb4"]) < (np.where(data["flux_d0_pb2"] > -1, data["flux_d0_pb2"], data["flux_d0_pb4"] )))*1.)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["distmod"]<0, data["1__fft_coefficient__coeff_0__attr__abs__y"], (-1.0*((np.where(data["hostgal_photoz_err"]<0, data["1__fft_coefficient__coeff_0__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] )))) )) +
                0.100000*np.tanh(((data["4__skewness_x"]) + (np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"]<0, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["distmod"] )))) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["1__kurtosis_y"], data["flux_err_mean"] )) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, data["distmod"], np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, ((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (data["1__fft_coefficient__coeff_0__attr__abs__x"])), data["1__fft_coefficient__coeff_1__attr__abs__x"] ) )))

    def GP_class_92(self,data):
        return (-1.730312 +
                0.100000*np.tanh(((data["flux_err_min"]) + (((((data["flux_err_min"]) + (-3.0))) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))))) +
                0.100000*np.tanh(((((data["detected_mean"]) + (((data["detected_mean"]) + (-2.0))))) + (-2.0))) +
                0.100000*np.tanh(((((np.minimum(((((((data["flux_err_min"]) + (data["detected_mean"]))) * 2.0))), ((data["detected_mean"])))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["2__fft_coefficient__coeff_1__attr__abs__y"])), ((((((np.minimum(((np.minimum(((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0))), ((data["flux_err_min"]))))), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) * 2.0)) * 2.0))))) +
                0.100000*np.tanh(np.where(((data["4__kurtosis_x"]) * 2.0) > -1, (-1.0*((data["4__kurtosis_x"]))), ((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0)) * 2.0) )) +
                0.100000*np.tanh((((-1.0*((data["5__kurtosis_x"])))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["detected_mean"])), ((((np.minimum(((np.minimum(((((data["flux_err_min"]) * 2.0))), ((data["detected_mean"]))))), ((data["flux_err_min"])))) * 2.0))))) +
                0.100000*np.tanh((((((np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__y"])), ((data["1__skewness_y"])))) + (-1.0))/2.0)) + (data["flux_err_min"]))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (-2.0))) + (((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__y"]))))))) + (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((np.minimum(((((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)) * 2.0))), ((((data["flux_err_min"]) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((data["5__kurtosis_x"]) + (((((((-1.0) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["4__kurtosis_y"]))))) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"] > -1, np.where(data["3__kurtosis_x"] > -1, -2.0, np.where(data["3__kurtosis_x"] > -1, data["detected_flux_diff"], data["detected_flux_std"] ) ), data["0__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(((((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["3__kurtosis_y"]))) * 2.0)) - (np.tanh((data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh((-1.0*((np.where(np.minimum(((data["3__skewness_x"])), ((data["3__kurtosis_x"]))) > -1, np.where(data["3__kurtosis_x"] > -1, 3.141593, data["3__kurtosis_x"] ), data["3__kurtosis_x"] ))))) +
                0.100000*np.tanh(np.where(-3.0<0, np.where(data["4__kurtosis_x"] > -1, ((-3.0) - (-2.0)), data["detected_mean"] ), data["flux_max"] )) +
                0.100000*np.tanh(((((np.minimum(((((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (data["flux_diff"])))), ((-3.0)))) + (np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__y"])), ((data["detected_flux_max"])))))) - (data["5__kurtosis_x"]))) +
                0.100000*np.tanh((-1.0*((np.where(np.where(data["flux_err_max"] > -1, data["4__kurtosis_x"], 2.718282 ) > -1, 2.0, data["5__kurtosis_x"] ))))) +
                0.100000*np.tanh(((np.minimum(((data["detected_flux_max"])), ((np.minimum(((data["detected_flux_err_min"])), ((data["0__fft_coefficient__coeff_1__attr__abs__x"]))))))) * 2.0)) +
                0.100000*np.tanh(np.minimum((((-1.0*((data["3__kurtosis_x"]))))), ((((((data["flux_err_min"]) * 2.0)) * 2.0))))) +
                0.100000*np.tanh(((((((np.minimum(((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0))), ((data["detected_flux_err_min"])))) * 2.0)) * 2.0)) + (data["detected_flux_err_min"]))) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), ((np.minimum(((data["detected_flux_max"])), ((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0))))))))), ((data["flux_err_min"])))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["0__fft_coefficient__coeff_1__attr__abs__y"]<0, np.where(((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)>0, data["flux_err_min"], data["1__fft_coefficient__coeff_1__attr__abs__x"] ), ((data["flux_err_min"]) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"] > -1, -3.0, np.where(data["3__kurtosis_y"] > -1, -3.0, ((data["0__fft_coefficient__coeff_1__attr__abs__y"]) - (-1.0)) ) )) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"] > -1, np.minimum(((data["5__kurtosis_x"])), ((-3.0))), np.where(2.0 > -1, data["3__fft_coefficient__coeff_1__attr__abs__x"], data["0__fft_coefficient__coeff_0__attr__abs__y"] ) )) +
                0.100000*np.tanh(np.where(data["5__kurtosis_x"] > -1, np.where(data["4__kurtosis_x"] > -1, -2.0, data["0__fft_coefficient__coeff_1__attr__abs__x"] ), data["detected_flux_std"] )) +
                0.100000*np.tanh(np.where(data["3__skewness_x"]<0, data["flux_diff"], np.minimum(((-2.0)), ((np.minimum(((np.where(data["5__fft_coefficient__coeff_0__attr__abs__y"]<0, data["flux_diff"], data["3__skewness_x"] ))), ((data["distmod"])))))) )) +
                0.100000*np.tanh(np.minimum(((data["flux_err_min"])), ((np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__y"])), ((((data["flux_max"]) * 2.0)))))))) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz"] > -1, np.where(data["hostgal_photoz"] > -1, -3.0, data["0__fft_coefficient__coeff_0__attr__abs__x"] ), data["0__fft_coefficient__coeff_0__attr__abs__y"] )) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_min"] > -1, -3.0, (-1.0*((data["5__kurtosis_y"]))) )) +
                0.100000*np.tanh(np.minimum(((np.minimum(((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (((data["flux_d1_pb2"]) * 2.0))))), ((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), ((data["2__fft_coefficient__coeff_1__attr__abs__y"])))))))), ((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0))))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (np.maximum(((data["4__kurtosis_x"])), ((data["detected_flux_min"])))))) - (data["1__kurtosis_y"]))) +
                0.100000*np.tanh(np.minimum(((((data["hostgal_photoz_err"]) + (np.minimum((((((data["flux_err_min"]) + (data["flux_err_min"]))/2.0))), ((np.minimum(((data["detected_flux_err_skew"])), ((data["1__fft_coefficient__coeff_1__attr__abs__x"])))))))))), ((data["detected_flux_max"])))) +
                0.100000*np.tanh(np.where(data["1__skewness_y"]>0, data["1__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["5__kurtosis_x"] > -1, -3.0, data["flux_err_min"] ) )) +
                0.100000*np.tanh(np.minimum(((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) - (np.maximum(((((data["detected_mean"]) - (data["1__fft_coefficient__coeff_0__attr__abs__x"])))), ((1.0))))))), ((data["flux_err_min"])))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["2__fft_coefficient__coeff_1__attr__abs__x"])), ((np.minimum(((np.minimum(((data["detected_mean"])), ((data["detected_flux_max"]))))), ((data["flux_err_skew"])))))))), ((data["detected_flux_err_min"])))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) - (data["4__kurtosis_y"]))) + (((data["0__fft_coefficient__coeff_0__attr__abs__x"]) - (data["flux_dif3"]))))) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_max"])), ((data["5__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh(np.where(data["4__skewness_x"]<0, np.where(data["detected_flux_max"]<0, data["detected_flux_err_min"], data["3__fft_coefficient__coeff_1__attr__abs__y"] ), -2.0 )) +
                0.100000*np.tanh(((((((-2.0) + (data["detected_flux_err_min"]))) + ((((data["2__skewness_x"]) + (data["1__skewness_y"]))/2.0)))) + (data["detected_flux_err_min"]))) +
                0.100000*np.tanh(np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__x"])), ((np.minimum(((data["flux_err_min"])), ((((data["detected_flux_max"]) + (data["2__skewness_y"]))))))))) +
                0.100000*np.tanh(np.minimum(((data["flux_err_min"])), ((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) - (data["4__kurtosis_x"])))))) +
                0.100000*np.tanh((((((-1.0*((((np.where(data["4__kurtosis_x"] > -1, data["5__kurtosis_x"], 1.0 )) * (data["1__fft_coefficient__coeff_0__attr__abs__x"])))))) - (data["5__kurtosis_y"]))) - (data["flux_min"]))) +
                0.100000*np.tanh(((data["flux_dif3"]) + ((-1.0*((((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) / 2.0)) + ((((data["flux_err_median"]) > (data["2__fft_coefficient__coeff_1__attr__abs__y"]))*1.))))))))) +
                0.100000*np.tanh(((data["flux_err_min"]) + (data["detected_flux_err_min"]))) +
                0.100000*np.tanh(((((data["detected_mean"]) - (np.where(((data["4__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0) > -1, data["4__fft_coefficient__coeff_1__attr__abs__x"], data["4__fft_coefficient__coeff_1__attr__abs__x"] )))) * 2.0)) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, -2.0, data["1__kurtosis_y"] )) +
                0.100000*np.tanh(np.where((-1.0*((((data["detected_flux_dif2"]) / 2.0)))) > -1, data["1__fft_coefficient__coeff_0__attr__abs__x"], data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(data["flux_max"]<0, (-1.0*(((-1.0*((data["2__fft_coefficient__coeff_0__attr__abs__x"])))))), np.where(data["4__skewness_x"]<0, (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["1__skewness_y"]))/2.0), -3.0 ) )) +
                0.100000*np.tanh((((((((np.minimum(((data["1__skewness_y"])), ((data["0__fft_coefficient__coeff_1__attr__abs__x"])))) + (data["detected_flux_max"]))/2.0)) - (data["2__kurtosis_y"]))) * 2.0)) +
                0.100000*np.tanh(np.where(np.minimum(((((data["flux_max"]) * (data["1__skewness_y"])))), ((data["detected_flux_std"])))>0, data["0__fft_coefficient__coeff_1__attr__abs__x"], np.minimum(((data["detected_flux_max"])), ((data["5__fft_coefficient__coeff_1__attr__abs__y"]))) )) +
                0.100000*np.tanh(((data["flux_err_skew"]) - (data["4__kurtosis_x"]))) +
                0.100000*np.tanh(np.where(data["flux_err_skew"]>0, data["flux_err_skew"], -3.0 )) +
                0.100000*np.tanh(((((data["detected_flux_by_flux_ratio_sq_skew"]) * 2.0)) - (data["3__kurtosis_x"]))) +
                0.100000*np.tanh((-1.0*((np.where(((data["detected_flux_min"]) - (data["5__kurtosis_x"]))>0, np.where(data["detected_flux_min"]<0, data["detected_flux_skew"], data["detected_flux_min"] ), data["5__kurtosis_x"] ))))) +
                0.100000*np.tanh(((data["ddf"]) - (data["4__kurtosis_x"]))) +
                0.100000*np.tanh(np.where(data["5__kurtosis_x"] > -1, -2.0, np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["5__fft_coefficient__coeff_0__attr__abs__y"], data["detected_flux_ratio_sq_sum"] ) )) +
                0.100000*np.tanh(((np.where(data["0__skewness_x"]>0, data["1__fft_coefficient__coeff_0__attr__abs__x"], ((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (data["4__fft_coefficient__coeff_0__attr__abs__y"])) )) - (data["4__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(np.minimum(((np.where(data["flux_ratio_sq_sum"] > -1, data["detected_flux_err_skew"], data["2__skewness_x"] ))), ((data["detected_flux_err_min"])))) +
                0.100000*np.tanh(((data["detected_flux_ratio_sq_skew"]) - (data["3__skewness_x"]))) +
                0.100000*np.tanh(np.minimum(((data["5__fft_coefficient__coeff_0__attr__abs__x"])), ((np.where(data["5__kurtosis_x"] > -1, np.minimum(((3.0)), ((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (data["mwebv"]))))), ((data["detected_mjd_size"]) / 2.0) ))))) +
                0.100000*np.tanh(np.minimum(((data["flux_dif3"])), ((np.minimum(((data["flux_dif3"])), ((data["detected_flux_err_std"]))))))) +
                0.100000*np.tanh(((((((data["1__skewness_y"]) + (np.where(data["1__skewness_y"] > -1, data["flux_err_std"], data["1__fft_coefficient__coeff_0__attr__abs__x"] )))) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) + (data["1__skewness_y"]))) +
                0.100000*np.tanh((((((data["flux_d0_pb5"]) < (np.minimum(((data["flux_max"])), ((data["5__fft_coefficient__coeff_0__attr__abs__x"])))))*1.)) / 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["flux_err_median"])), ((((((((data["1__kurtosis_y"]) < (data["2__fft_coefficient__coeff_1__attr__abs__x"]))*1.)) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))/2.0))))) - (data["4__skewness_y"]))) +
                0.100000*np.tanh(((((data["2__skewness_y"]) * (((data["1__fft_coefficient__coeff_1__attr__abs__y"]) * (data["2__skewness_y"]))))) - ((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (((((data["flux_d1_pb3"]) / 2.0)) * 2.0)))/2.0)))) +
                0.100000*np.tanh(np.where(np.where(data["2__fft_coefficient__coeff_0__attr__abs__y"]>0, np.where((9.86017036437988281)>0, data["flux_d0_pb3"], data["flux_dif3"] ), (-1.0*((data["4__skewness_y"]))) ) > -1, data["detected_flux_err_skew"], data["2__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(np.minimum(((((-1.0) / 2.0))), ((np.tanh((data["detected_mean"])))))) +
                0.100000*np.tanh((((((np.tanh((data["1__skewness_y"]))) - (data["flux_d1_pb3"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) +
                0.100000*np.tanh(((np.minimum(((np.tanh((data["0__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["2__skewness_y"])))) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_d1_pb5"] > -1, np.where(data["detected_flux_min"] > -1, data["detected_flux_min"], data["0__skewness_x"] ), data["detected_flux_err_skew"] )) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"]<0, data["1__fft_coefficient__coeff_1__attr__abs__x"], ((data["flux_dif3"]) + (data["mwebv"])) )) +
                0.100000*np.tanh(((((np.where(data["4__skewness_x"]<0, data["1__skewness_y"], (((-1.0*((((data["2__fft_coefficient__coeff_0__attr__abs__x"]) / 2.0))))) / 2.0) )) * 2.0)) / 2.0)) +
                0.100000*np.tanh(((data["detected_mean"]) - (np.where(data["flux_dif3"] > -1, data["1__fft_coefficient__coeff_0__attr__abs__x"], (((data["3__kurtosis_x"]) < (data["2__kurtosis_y"]))*1.) )))) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb4"]>0, np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), ((data["flux_dif3"]))), data["flux_d0_pb4"] )) - (data["detected_flux_mean"]))) +
                0.100000*np.tanh(np.where(np.maximum((((((((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)) + (data["detected_flux_by_flux_ratio_sq_sum"])))), ((data["2__fft_coefficient__coeff_1__attr__abs__y"]))) > -1, data["0__fft_coefficient__coeff_0__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb1"]>0, ((data["detected_flux_diff"]) + (np.where(data["flux_dif3"]>0, np.maximum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), ((data["detected_flux_err_skew"]))), -3.0 ))), data["flux_d1_pb1"] )) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]<0, (((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) + (data["5__skewness_x"]))/2.0), data["mwebv"] )) +
                0.100000*np.tanh(((np.minimum(((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["1__kurtosis_x"])))), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) + (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0)))) +
                0.100000*np.tanh(np.where((-1.0*((np.tanh((data["detected_flux_std"])))))<0, data["1__fft_coefficient__coeff_0__attr__abs__y"], data["4__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(np.where((((np.where(data["flux_ratio_sq_sum"] > -1, data["3__fft_coefficient__coeff_0__attr__abs__x"], data["detected_flux_by_flux_ratio_sq_skew"] )) + (data["detected_flux_by_flux_ratio_sq_skew"]))/2.0)<0, data["flux_d1_pb4"], data["mwebv"] )) +
                0.100000*np.tanh(((((data["detected_flux_err_mean"]) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__skewness_x"]))) +
                0.100000*np.tanh(np.minimum(((np.where(data["1__skewness_y"]<0, np.where(data["detected_mean"]>0, data["1__skewness_y"], data["1__skewness_y"] ), data["1__fft_coefficient__coeff_1__attr__abs__y"] ))), ((data["flux_d1_pb1"])))) +
                0.100000*np.tanh(np.where(data["detected_flux_max"]>0, ((((data["detected_flux_by_flux_ratio_sq_skew"]) + (data["1__skewness_y"]))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"])), data["2__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(((data["detected_flux_err_skew"]) - (np.where(((data["flux_diff"]) / 2.0) > -1, ((data["flux_diff"]) - (data["flux_d1_pb4"])), ((data["flux_diff"]) / 2.0) )))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (np.tanh((data["2__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(np.where(data["detected_flux_min"] > -1, data["flux_by_flux_ratio_sq_skew"], np.where(data["5__kurtosis_x"] > -1, -1.0, data["detected_flux_diff"] ) )) +
                0.100000*np.tanh(((data["detected_flux_ratio_sq_skew"]) * (2.0))) +
                0.100000*np.tanh(((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) * (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) + (np.maximum(((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_d0_pb3"])))), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))))) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]<0, np.minimum(((data["flux_err_skew"])), ((((data["detected_flux_err_max"]) - (data["2__kurtosis_y"]))))), data["flux_ratio_sq_sum"] )) +
                0.100000*np.tanh(np.minimum(((data["0__kurtosis_y"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh((-1.0*((np.where(data["flux_ratio_sq_skew"]>0, np.tanh((data["0__fft_coefficient__coeff_1__attr__abs__y"])), (-1.0*((data["0__fft_coefficient__coeff_1__attr__abs__x"]))) ))))) +
                0.100000*np.tanh(np.where((((data["flux_err_skew"]) > (((data["detected_flux_err_skew"]) / 2.0)))*1.)<0, data["1__fft_coefficient__coeff_0__attr__abs__x"], np.minimum(((data["5__fft_coefficient__coeff_0__attr__abs__x"])), ((data["detected_flux_err_skew"]))) )) +
                0.100000*np.tanh(np.minimum(((data["flux_err_max"])), ((((data["flux_err_skew"]) * (((data["4__fft_coefficient__coeff_1__attr__abs__x"]) + (-3.0)))))))) +
                0.100000*np.tanh((((((2.718282) / 2.0)) < (np.where(data["mwebv"]<0, data["flux_ratio_sq_sum"], (-1.0*(((-1.0*((data["mjd_size"])))))) )))*1.)) +
                0.100000*np.tanh((((data["flux_err_skew"]) + (data["detected_mean"]))/2.0)) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_1__attr__abs__y"] > -1, np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__x"])), (((((np.where(data["flux_max"]<0, data["2__fft_coefficient__coeff_1__attr__abs__y"], data["flux_dif3"] )) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))/2.0)))), -3.0 )) +
                0.100000*np.tanh(((np.minimum(((data["detected_mean"])), (((((((data["detected_flux_err_min"]) + (data["mwebv"]))/2.0)) / 2.0))))) / 2.0)) +
                0.100000*np.tanh(np.where(data["5__kurtosis_x"] > -1, (-1.0*(((((data["5__skewness_x"]) < (data["5__kurtosis_x"]))*1.)))), (((data["flux_d1_pb5"]) < (data["5__fft_coefficient__coeff_1__attr__abs__y"]))*1.) )) +
                0.100000*np.tanh((-1.0*((np.tanh(((-1.0*(((((((-1.0*((((np.tanh((data["1__skewness_y"]))) / 2.0))))) * (data["3__skewness_x"]))) * 2.0)))))))))) +
                0.100000*np.tanh(np.minimum(((data["1__skewness_x"])), ((data["0__kurtosis_y"])))) +
                0.100000*np.tanh(((((((data["flux_std"]) > (np.minimum(((data["5__skewness_y"])), ((data["flux_diff"])))))*1.)) < (np.minimum(((data["flux_err_min"])), ((data["1__skewness_x"])))))*1.)) +
                0.100000*np.tanh(((data["3__fft_coefficient__coeff_1__attr__abs__y"]) - ((-1.0*((data["4__kurtosis_y"])))))) +
                0.100000*np.tanh(((((((-1.0*((data["1__fft_coefficient__coeff_0__attr__abs__y"])))) + (data["4__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) - (((data["1__kurtosis_x"]) * 2.0)))) +
                0.100000*np.tanh(np.where((2.0)>0, (((((data["flux_d1_pb3"]) < (data["3__kurtosis_y"]))*1.)) / 2.0), (((((data["flux_err_std"]) / 2.0)) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))/2.0) )) +
                0.100000*np.tanh(((((np.tanh((np.tanh((((data["detected_flux_mean"]) + (data["detected_flux_by_flux_ratio_sq_skew"]))))))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh((((np.minimum(((0.0)), ((data["flux_err_mean"])))) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)) +
                0.100000*np.tanh(((np.where(data["detected_flux_err_skew"]<0, data["1__skewness_y"], data["flux_d1_pb0"] )) - (data["detected_flux_err_skew"]))) +
                0.100000*np.tanh(np.tanh((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) - (data["detected_flux_median"]))))) +
                0.100000*np.tanh((-1.0*((np.where(data["flux_max"] > -1, ((((data["mjd_diff"]) - (np.tanh((data["flux_max"]))))) / 2.0), data["flux_dif3"] ))))) +
                0.100000*np.tanh(np.minimum((((((data["0__skewness_y"]) < (((np.tanh((data["flux_dif3"]))) * 2.0)))*1.))), ((np.minimum(((-1.0)), ((data["1__skewness_x"]))))))) +
                0.100000*np.tanh(np.where(data["detected_flux_min"]>0, 2.0, data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_0__attr__abs__x"]<0, ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0), np.minimum(((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (data["1__fft_coefficient__coeff_0__attr__abs__x"])))), ((data["1__fft_coefficient__coeff_0__attr__abs__x"]))) )) +
                0.100000*np.tanh(np.where(data["detected_flux_min"]<0, np.where(data["4__kurtosis_x"]<0, data["detected_flux_by_flux_ratio_sq_skew"], ((-2.0) - (data["1__fft_coefficient__coeff_1__attr__abs__y"])) ), ((-2.0) - (data["detected_flux_min"])) )) +
                0.100000*np.tanh(((np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]>0, data["1__fft_coefficient__coeff_1__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__x"] )) / 2.0)) +
                0.100000*np.tanh(np.minimum((((-1.0*((data["3__fft_coefficient__coeff_1__attr__abs__y"]))))), ((data["flux_d0_pb4"])))) +
                0.100000*np.tanh(np.where(data["detected_flux_by_flux_ratio_sq_skew"] > -1, data["flux_max"], ((data["flux_max"]) - (((((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (data["1__skewness_y"]))) / 2.0))) )) +
                0.100000*np.tanh(np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__y"])), (((((3.0)) * 2.0))))) +
                0.100000*np.tanh(((((3.141593) - (((((data["flux_ratio_sq_sum"]) * 2.0)) - (data["mwebv"]))))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_flux_err_skew"]))) +
                0.100000*np.tanh(np.where((((data["1__skewness_y"]) < (data["flux_err_skew"]))*1.) > -1, data["1__fft_coefficient__coeff_1__attr__abs__y"], data["hostgal_photoz"] )))

    def GP_class_95(self,data):
        return (-1.890339 +
                0.100000*np.tanh(((((((((data["hostgal_photoz"]) * 2.0)) + (np.tanh((data["0__fft_coefficient__coeff_1__attr__abs__y"]))))) * 2.0)) + (data["distmod"]))) +
                0.100000*np.tanh(((((((data["distmod"]) + (data["hostgal_photoz"]))) + (data["distmod"]))) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(((((((((((data["hostgal_photoz"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["distmod"]) + (data["distmod"]))) +
                0.100000*np.tanh(((((((data["hostgal_photoz"]) + (data["hostgal_photoz"]))) * 2.0)) + (data["flux_w_mean"]))) +
                0.100000*np.tanh(((((((data["distmod"]) + (((data["distmod"]) * 2.0)))) * 2.0)) + (((data["hostgal_photoz"]) + (data["hostgal_photoz"]))))) +
                0.100000*np.tanh(((((((data["distmod"]) + (((data["distmod"]) * 2.0)))) + (data["distmod"]))) + (((data["distmod"]) + (data["detected_flux_w_mean"]))))) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) + (data["hostgal_photoz"]))) * 2.0)) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (((data["hostgal_photoz"]) + (((data["distmod"]) + (((data["hostgal_photoz"]) + (data["distmod"]))))))))) +
                0.100000*np.tanh(((((data["flux_d1_pb4"]) + (((data["detected_flux_min"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (((data["distmod"]) + (((((np.minimum(((data["3__skewness_y"])), ((((data["flux_d1_pb0"]) * 2.0))))) / 2.0)) + (data["hostgal_photoz"]))))))) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) + (data["hostgal_photoz"]))) + (((((data["hostgal_photoz"]) + (data["hostgal_photoz"]))) * 2.0)))) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (np.minimum(((((data["flux_d1_pb5"]) + (data["hostgal_photoz_err"])))), ((data["distmod"])))))) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (((data["flux_d0_pb5"]) + (((data["hostgal_photoz"]) + (data["flux_d0_pb5"]))))))) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (((data["hostgal_photoz"]) + (data["distmod"]))))) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) + (((data["flux_median"]) + (data["hostgal_photoz"]))))) * 2.0)) +
                0.100000*np.tanh((((((((((data["hostgal_photoz"]) * 2.0)) * 2.0)) + (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (data["hostgal_photoz"]))))/2.0)) + (data["flux_d0_pb3"]))) +
                0.100000*np.tanh(np.minimum(((data["flux_d0_pb5"])), ((np.minimum(((data["distmod"])), ((((data["flux_max"]) + (data["5__skewness_x"]))))))))) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["distmod"]) + ((((data["0__skewness_x"]) + (((data["distmod"]) * 2.0)))/2.0)))))) +
                0.100000*np.tanh(((((np.minimum(((((((data["hostgal_photoz"]) + (data["flux_d0_pb1"]))) * 2.0))), ((data["hostgal_photoz"])))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (((((np.minimum(((data["hostgal_photoz"])), ((data["5__skewness_x"])))) * 2.0)) + (data["flux_d0_pb5"]))))) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(((((data["flux_median"]) + (np.minimum(((np.minimum(((data["hostgal_photoz"])), ((data["hostgal_photoz"]))))), ((((((data["hostgal_photoz"]) * 2.0)) + (data["detected_flux_min"])))))))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((((data["distmod"]) + (((((data["distmod"]) * 2.0)) * 2.0))))), ((((data["flux_d0_pb5"]) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((((((((data["flux_d0_pb4"]) - (data["hostgal_photoz_err"]))) * 2.0)) + (((data["flux_d0_pb4"]) - (data["flux_d0_pb5"]))))) * 2.0)) +
                0.100000*np.tanh(((data["flux_d1_pb4"]) + (((data["detected_flux_min"]) + (data["4__skewness_x"]))))) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) * 2.0)) + (data["detected_flux_std"]))) +
                0.100000*np.tanh(((((data["distmod"]) - (((data["hostgal_photoz_err"]) * 2.0)))) + (data["0__skewness_x"]))) +
                0.100000*np.tanh(((np.minimum(((data["flux_d0_pb5"])), ((data["hostgal_photoz"])))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((np.minimum(((data["hostgal_photoz"])), ((((data["hostgal_photoz"]) * 2.0)))))), ((data["4__fft_coefficient__coeff_1__attr__abs__y"]))))), ((((data["hostgal_photoz"]) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["distmod"])), ((np.minimum(((data["1__skewness_x"])), ((((((data["flux_d1_pb5"]) * 2.0)) + (((data["1__skewness_x"]) * 2.0)))))))))) * 2.0)) +
                0.100000*np.tanh(((((((data["5__skewness_x"]) + (data["detected_flux_min"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["distmod"])), ((((data["0__skewness_x"]) + (data["0__skewness_x"])))))) + (((data["0__skewness_x"]) + (((data["0__skewness_x"]) * 2.0)))))) +
                0.100000*np.tanh(((data["distmod"]) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((((data["detected_flux_median"]) + (data["hostgal_photoz"]))) + (data["flux_std"]))) + (((np.minimum(((data["5__skewness_x"])), ((data["hostgal_photoz"])))) + (data["hostgal_photoz"]))))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((np.minimum(((data["4__skewness_x"])), ((data["hostgal_photoz"])))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((((data["distmod"]) - (data["hostgal_photoz_err"])))), ((((data["hostgal_photoz"]) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_d0_pb5"]) - (data["hostgal_photoz_err"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (data["hostgal_photoz"]))) + (data["flux_std"]))) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_1__attr__abs__y"]) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((((data["flux_d0_pb5"]) - (data["hostgal_photoz_err"]))) - (data["hostgal_photoz_err"]))) * 2.0)) +
                0.100000*np.tanh(((((((((data["hostgal_photoz"]) + (data["5__fft_coefficient__coeff_0__attr__abs__x"]))) + (((data["hostgal_photoz"]) + (data["hostgal_photoz"]))))) + (data["hostgal_photoz"]))) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(np.minimum(((data["5__skewness_x"])), ((((((data["0__skewness_x"]) * 2.0)) * 2.0))))) +
                0.100000*np.tanh(((((data["4__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["distmod"]) + (data["detected_flux_min"]))))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((((((data["detected_flux_err_mean"]) + (data["hostgal_photoz"]))) + (data["hostgal_photoz"])))), ((data["flux_d0_pb5"])))) + (((data["hostgal_photoz"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["hostgal_photoz"])), ((((data["flux_d0_pb4"]) - (data["hostgal_photoz_err"]))))))), ((data["1__skewness_x"])))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz_err"]<0, data["distmod"], ((((data["hostgal_photoz"]) - (data["hostgal_photoz_err"]))) - (data["hostgal_photoz_err"])) )) + (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_max"]))) + (((((data["hostgal_photoz"]) + (data["flux_std"]))) + (((data["hostgal_photoz"]) + (-1.0))))))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (np.where(((data["2__kurtosis_x"]) / 2.0) > -1, data["hostgal_photoz_err"], (((((data["hostgal_photoz_err"]) * 2.0)) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))/2.0) )))) +
                0.100000*np.tanh(((((data["flux_d1_pb5"]) + (np.minimum(((data["1__skewness_x"])), ((np.minimum(((data["flux_d0_pb4"])), ((np.minimum(((data["detected_mjd_diff"])), ((data["flux_skew"])))))))))))) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((data["flux_diff"]) + (((data["hostgal_photoz"]) + (((np.minimum(((data["flux_diff"])), ((data["5__fft_coefficient__coeff_0__attr__abs__x"])))) * 2.0)))))) + (((data["hostgal_photoz"]) * 2.0)))) +
                0.100000*np.tanh(((np.minimum(((((np.minimum(((data["detected_flux_min"])), ((((np.tanh((data["hostgal_photoz"]))) - (data["hostgal_photoz_err"])))))) * 2.0))), ((data["hostgal_photoz"])))) * 2.0)) +
                0.100000*np.tanh(((((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["2__kurtosis_x"]))) - (data["2__kurtosis_x"]))) - (data["2__kurtosis_x"]))) +
                0.100000*np.tanh(((((np.minimum(((data["flux_skew"])), ((((((data["hostgal_photoz"]) * 2.0)) * 2.0))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["hostgal_photoz_err"]) + (data["hostgal_photoz"]))) - (((data["hostgal_photoz_err"]) * (data["hostgal_photoz"]))))) +
                0.100000*np.tanh(((np.where(data["3__skewness_x"]<0, data["flux_dif3"], ((data["flux_d0_pb0"]) + (data["4__fft_coefficient__coeff_0__attr__abs__x"])) )) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (((data["hostgal_photoz_err"]) + (((data["flux_d1_pb0"]) + (np.minimum(((data["hostgal_photoz"])), ((data["detected_mjd_diff"])))))))))) +
                0.100000*np.tanh(np.where(data["flux_dif2"]<0, data["flux_d1_pb4"], -3.0 )) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) + (((((((data["hostgal_photoz"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)) + (data["hostgal_photoz"]))))) * 2.0)) +
                0.100000*np.tanh(((((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["hostgal_photoz_err"]))) - (data["1__skewness_y"]))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((((np.minimum(((data["flux_median"])), ((((data["flux_d0_pb5"]) - (data["flux_skew"])))))) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((((((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["hostgal_photoz"]))) * 2.0)) + (0.0))) * 2.0)) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["2__kurtosis_y"]) - (data["hostgal_photoz_err"]))) * 2.0)) +
                0.100000*np.tanh(((((np.where(data["flux_skew"] > -1, data["flux_skew"], data["hostgal_photoz"] )) - (((data["hostgal_photoz_err"]) * (data["flux_skew"]))))) * (data["hostgal_photoz"]))) +
                0.100000*np.tanh(np.minimum(((data["flux_median"])), ((((data["1__skewness_x"]) * 2.0))))) +
                0.100000*np.tanh(((((data["flux_diff"]) + (data["detected_flux_err_mean"]))) + (data["3__skewness_x"]))) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (((((-1.0) + (((data["hostgal_photoz"]) * 2.0)))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(((np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]>0, ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["2__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0))) )) * 2.0)) +
                0.100000*np.tanh(((((((((data["detected_flux_median"]) + (data["4__fft_coefficient__coeff_0__attr__abs__x"]))) * 2.0)) / 2.0)) + (data["0__skewness_x"]))) +
                0.100000*np.tanh((((((data["hostgal_photoz_err"]) < (data["4__fft_coefficient__coeff_1__attr__abs__y"]))*1.)) - ((((data["hostgal_photoz_err"]) + (np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"] > -1, data["hostgal_photoz_err"], data["4__fft_coefficient__coeff_1__attr__abs__y"] )))/2.0)))) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (np.minimum(((((data["hostgal_photoz"]) + (((data["hostgal_photoz"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"])))))), ((data["flux_max"])))))) +
                0.100000*np.tanh(np.minimum(((np.tanh((np.minimum(((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) - (data["hostgal_photoz_err"])))), ((data["flux_median"]))))))), ((data["flux_median"])))) +
                0.100000*np.tanh((((((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (np.where(data["5__fft_coefficient__coeff_0__attr__abs__y"]<0, data["1__kurtosis_x"], data["5__fft_coefficient__coeff_0__attr__abs__y"] )))/2.0)) - (data["hostgal_photoz_err"]))) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["detected_mjd_diff"])), ((data["detected_mjd_diff"]))))), ((np.where(data["flux_median"]<0, data["detected_mjd_diff"], np.minimum(((data["detected_mjd_diff"])), ((data["1__fft_coefficient__coeff_1__attr__abs__x"]))) ))))) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) + (data["detected_flux_mean"]))) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(((np.minimum(((data["flux_d0_pb0"])), ((np.minimum(((((np.minimum(((data["detected_mjd_size"])), ((data["3__fft_coefficient__coeff_0__attr__abs__y"])))) / 2.0))), ((data["1__skewness_x"]))))))) * 2.0)) +
                0.100000*np.tanh(((((data["2__kurtosis_y"]) + ((((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (data["hostgal_photoz"]))/2.0)))) + (((data["hostgal_photoz"]) + (data["hostgal_photoz"]))))) +
                0.100000*np.tanh(np.minimum(((data["2__kurtosis_y"])), ((np.minimum(((data["2__kurtosis_y"])), ((np.minimum(((np.minimum(((data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((data["flux_d0_pb3"]))))), ((((((data["flux_median"]) * 2.0)) * 2.0))))))))))) +
                0.100000*np.tanh(((((((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (data["hostgal_photoz"]))) + (data["5__fft_coefficient__coeff_0__attr__abs__x"]))) + (((((((data["hostgal_photoz"]) * 2.0)) * 2.0)) + (data["hostgal_photoz_err"]))))) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + ((((data["0__skewness_y"]) + (((data["detected_flux_err_mean"]) + (((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["hostgal_photoz"]) + (data["5__fft_coefficient__coeff_1__attr__abs__y"]))))))))/2.0)))) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz"]>0, ((np.where(data["hostgal_photoz"] > -1, data["4__fft_coefficient__coeff_1__attr__abs__y"], data["detected_flux_err_median"] )) - (data["hostgal_photoz_err"])), data["0__fft_coefficient__coeff_0__attr__abs__y"] )) - (data["4__skewness_x"]))) +
                0.100000*np.tanh(((data["2__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_min"]))) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (((((np.where(data["flux_skew"]<0, data["detected_flux_err_std"], data["hostgal_photoz"] )) + (data["4__skewness_y"]))) + (data["flux_d0_pb0"]))))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) + (data["hostgal_photoz"]))) + (((((data["1__skewness_y"]) + (data["hostgal_photoz"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(np.where(data["2__kurtosis_x"]>0, data["mjd_diff"], np.where(data["2__kurtosis_x"]>0, ((data["3__kurtosis_y"]) + (data["2__kurtosis_x"])), data["3__kurtosis_y"] ) )) +
                0.100000*np.tanh(np.where(data["1__skewness_x"]<0, data["1__skewness_x"], data["detected_flux_mean"] )) +
                0.100000*np.tanh(((data["flux_d1_pb3"]) + (((np.minimum(((data["2__kurtosis_y"])), ((data["detected_flux_max"])))) + (data["2__kurtosis_y"]))))) +
                0.100000*np.tanh(((((((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["hostgal_photoz"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_min"] > -1, ((((data["flux_min"]) / 2.0)) - (data["1__skewness_y"])), data["1__skewness_x"] )) +
                0.100000*np.tanh(np.minimum(((data["flux_median"])), ((((data["flux_skew"]) + (np.minimum(((np.minimum(((data["flux_median"])), ((np.minimum(((data["detected_mjd_diff"])), ((data["detected_mjd_diff"])))))))), ((data["flux_median"]))))))))) +
                0.100000*np.tanh((((data["flux_d0_pb5"]) + (data["flux_skew"]))/2.0)) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) + (data["detected_flux_err_median"]))) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_ratio_sq_skew"]<0, data["flux_d1_pb1"], ((np.minimum(((data["detected_mjd_diff"])), ((data["5__fft_coefficient__coeff_0__attr__abs__y"])))) - (data["detected_mean"])) )) +
                0.100000*np.tanh(np.where(data["detected_flux_ratio_sq_skew"]<0, (((((data["flux_d0_pb5"]) * 2.0)) + (data["1__skewness_x"]))/2.0), data["detected_mjd_diff"] )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz_err"]<0, data["3__skewness_x"], np.where(data["hostgal_photoz"]<0, np.maximum(((data["hostgal_photoz"])), ((data["hostgal_photoz_err"]))), ((data["3__skewness_x"]) - (data["hostgal_photoz_err"])) ) )) +
                0.100000*np.tanh((((data["flux_median"]) + (((data["detected_flux_err_median"]) + (data["4__fft_coefficient__coeff_1__attr__abs__x"]))))/2.0)) +
                0.100000*np.tanh(np.where(data["flux_dif3"]>0, np.where(np.where(data["flux_dif3"]>0, data["0__fft_coefficient__coeff_0__attr__abs__y"], data["flux_dif3"] )>0, data["flux_dif3"], data["flux_dif3"] ), data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(((((((((((data["distmod"]) - (data["hostgal_photoz_err"]))) - (data["1__skewness_y"]))) - (data["1__skewness_y"]))) - (data["1__skewness_y"]))) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((np.where(((data["3__kurtosis_y"]) * 2.0)<0, data["5__fft_coefficient__coeff_0__attr__abs__x"], np.tanh((((data["1__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0))) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["flux_d1_pb0"]) + (data["hostgal_photoz"]))) + (((data["hostgal_photoz"]) + (data["hostgal_photoz"]))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["detected_mjd_diff"])), ((data["2__kurtosis_y"]))))), ((data["2__kurtosis_y"])))) +
                0.100000*np.tanh(((data["detected_flux_err_std"]) + (data["flux_d1_pb5"]))) +
                0.100000*np.tanh(np.where((-1.0*((data["hostgal_photoz_err"]))) > -1, data["hostgal_photoz"], (-1.0*((((data["detected_flux_err_min"]) + (data["hostgal_photoz"]))))) )) +
                0.100000*np.tanh(((data["hostgal_photoz_err"]) + (((((data["hostgal_photoz_err"]) + (((data["hostgal_photoz"]) + (((data["hostgal_photoz_err"]) + (data["flux_max"]))))))) + (data["hostgal_photoz"]))))) +
                0.100000*np.tanh(np.where(data["4__kurtosis_y"]>0, np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]>0, np.where(data["flux_err_std"]>0, data["flux_err_min"], data["0__fft_coefficient__coeff_0__attr__abs__y"] ), data["3__kurtosis_y"] ), data["flux_median"] )) +
                0.100000*np.tanh(((data["hostgal_photoz_err"]) + (((((data["flux_d1_pb0"]) - (data["hostgal_photoz_err"]))) - (data["hostgal_photoz_err"]))))) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) + (data["hostgal_photoz"]))) + (((data["detected_mean"]) + (((data["hostgal_photoz_err"]) + (data["detected_flux_skew"]))))))) +
                0.100000*np.tanh(np.where(data["2__kurtosis_x"]>0, np.tanh((np.where(((data["5__fft_coefficient__coeff_0__attr__abs__x"]) - ((-1.0*((data["4__fft_coefficient__coeff_1__attr__abs__y"])))))>0, data["mjd_diff"], data["1__kurtosis_y"] ))), data["4__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(((data["0__skewness_y"]) + (((data["3__skewness_x"]) + (np.minimum(((((data["4__fft_coefficient__coeff_0__attr__abs__x"]) + (data["flux_median"])))), ((data["0__skewness_y"])))))))) +
                0.100000*np.tanh(((data["flux_d1_pb5"]) + (((data["1__skewness_x"]) + (((((((data["1__kurtosis_x"]) + (data["1__skewness_x"]))) + (data["5__skewness_x"]))) - (data["hostgal_photoz_err"]))))))) +
                0.100000*np.tanh(((-1.0) + (((data["hostgal_photoz"]) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["hostgal_photoz"]))))))) +
                0.100000*np.tanh(((((data["detected_flux_skew"]) + (data["4__skewness_y"]))) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((data["flux_err_min"]) - (data["flux_err_min"]))) + (((data["flux_err_mean"]) * (data["1__skewness_y"]))))) +
                0.100000*np.tanh(((((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (np.where(data["5__skewness_x"]>0, data["detected_flux_max"], data["flux_d1_pb5"] )))) / 2.0)) + (data["1__kurtosis_x"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_err_std"]<0, data["flux_d0_pb5"], data["flux_d0_pb5"] )) +
                0.100000*np.tanh(np.minimum(((((data["3__kurtosis_y"]) - (np.minimum(((data["0__skewness_y"])), ((3.0))))))), ((data["3__fft_coefficient__coeff_1__attr__abs__x"])))) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz"] > -1, (-1.0*((data["4__kurtosis_x"]))), data["distmod"] )) + (data["distmod"]))) +
                0.100000*np.tanh(((((((data["hostgal_photoz"]) + (data["hostgal_photoz_err"]))) + (((data["hostgal_photoz"]) + (data["hostgal_photoz_err"]))))) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), ((np.minimum(((data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((data["detected_mjd_diff"]))))))))

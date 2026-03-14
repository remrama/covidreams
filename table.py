# import pickle


# def extract_regression_stats(filepath):
#     filepath = "../data/derivatives/regression-modl.pkl"
#     # filepath = "../data/derivatives/regression-stat.txt"
#     assert filepath.suffix == ".pkl"
#     with open(filepath, "rb") as f:
#         result = pickle.load(f)
#     b1_var = "Time"
#     b2_var = "Covid"
#     b3_var = "TimeCovid"
#     betas = result.params.iloc[1:].to_frame("beta")
#     pvals = result.pvalues.iloc[1:].to_frame("pval")
#     # tvals = result.tvalues.iloc[1:].to_frame("tval")
#     betas = betas.map(lambda x: f"{x:.2f}")
#     pvals = pvals.map(lambda x: f"{x:.3f}".lstrip("0")).replace(".000", "<.001")
#     civals = result.conf_int().rename(columns={0: "beta_lower", 1: "beta_upper"})

#     ndays = result.nobs
#     dof = int(result.df_model)
#     dof = int(result.df_resid)

#     df = betas.join(pvals).unstack(0).to_frame("dreams2019").T.swaplevel(axis=1).round(3)


# def extract_chi2_stats(filepath):
#     filepath = "../data/derivatives/chisquared-desc.tsv"
#     filepath = "../data/derivatives/chisquared-stat.tsv"
#     stat = pd.read_table(filepath, index_col="test")
#     desc = pd.read_table(filepath, index_col="PostCovid")
#     stat["p"].map("{:.3f}".format)

#     stat["lambda"] = stat["lambda"].map("{:.2f}".format)
#     stat["chi2"] = stat["chi2"].map("{:.2f}".format)
#     stat["dof"] = stat["dof"].astype(int).map("{:d}".format)
#     stat["pval"] = stat["pval"].map("{:.3f}".format).str.lstrip("0").replace(".000", "<.001")
#     stat["cramer"] = stat["cramer"].map("{:.2f}".format)
#     stat["power"] = stat["power"].map("{:.2f}".format)


# def extract_correlation_stats(filepath):
#     filepath = "../data/derivatives/correlation-stat.tsv"
#     stat = pd.read_table(filepath, index_col="method")
#     stat["r"] = stat["r"].map("{:.2f}".format)
#     stat["p_val"] = stat["p_val"].map("{:.3f}".format).str.lstrip("0").replace(".000", "<.001")
#     stat["power"] = stat["power"].map("{:.2f}".format)
#     stat = stat.rename(columns={"p_val": "pval"})
#     ci_idx = stat.columns.tolist().index("CI95")
#     ci = stat.pop("CI95").str.strip("[]").str.split()
#     r_lower = ci.str[0].astype(float).map("{:2f}".format)
#     r_upper = ci.str[1].astype(float).map("{:2f}".format)
#     stat.insert(2, "r_upper", r_upper)
#     stat.insert(2, "r_lower", r_lower)

#     filepath = "../data/derivatives/correlation-vals.tsv"
#     vals = pd.read_table(filepath)

#     filepath = "../data/derivatives/correlation-stat.tsv"
#     stat = pd.read_table(filepath, index_col="method")


# def extract_
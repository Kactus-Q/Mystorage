import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def ytm(p,t,coupon,n):
    yield11 = 0.0000001
    while abs(p + coupon * 100 * (182-t)/365 - ((coupon * 100 / yield11 * (1-(1/((1+yield11/2)**n))) + 100/((1+yield11/2)**n))*((1+yield11/2)**((182-t)/182.5)))) >= 0.0001:
        yield11 = yield11 + 0.0000001
    return yield11

def ytm1(p,t,coupon,n):    #for bond 11 only note that the issue date of bond11 is 10/11/2019, this date is later than 9/1/2019, 
                            # so the date used for accrued interest will be different from other bonds
    yield11 = 0.0000001
    while abs(p + coupon * 100 * (141-t)/365 - ((coupon * 100 / yield11 * (1-(1/((1+yield11/2)**n))) + 100/((1+yield11/2)**n))*((1+yield11/2)**((182-t)/182.5)))) >= 0.0001:
        yield11 = yield11 + 0.0000001
    return yield11

def ytm_for_eachday(bond1,bond2,bond3,bond4,bond5,bond6,bond7,bond8,bond9,bond10,bond11,date):
    ytm_list = [[],[],[],[],[],[],[],[],[],[]]
    Daycount1 = [59,58,55,54,53,52,51,48,47,46]
    Daycount2 = [151,150,147,146,145,144,143,140,139,138]   # number of days from settlement date to next coupon date

    for i in range(len(date)):
        #bond1
        p11 = float(bond1[date[i]])
        coupon1 = float(bond1['Coupon'])
        y11 = ytm(p11,Daycount1[i],coupon1,1)
        ytm_list[i].append(round(y11,4))

        #bond2
        p12 = float(bond2[date[i]])
        coupon2 = float(bond2['Coupon'])
        y12 = ytm(p12,Daycount1[i],coupon2,2)
        ytm_list[i].append(round(y12,4))

        #bond3
        p13 = float(bond3[date[i]])
        coupon3 = float(bond3['Coupon'])
        y13 = ytm(p13,Daycount1[i],coupon3,3)
        ytm_list[i].append(round(y13,4))

        #bond4
        p14 = float(bond4[date[i]])
        coupon4 = float(bond4['Coupon'])
        y14 = ytm(p14,Daycount1[i],coupon4,4)
        ytm_list[i].append(round(y14,4))

        #bond5
        p15 = float(bond5[date[i]])
        coupon5 = float(bond5['Coupon'])
        y15 = ytm(p15,Daycount1[i],coupon5,5)
        ytm_list[i].append(round(y15,4))

        #bond6
        p16 = float(bond6[date[i]])
        coupon6 = float(bond6['Coupon'])
        y16 = ytm(p16,Daycount2[i],coupon6,5)
        ytm_list[i].append(round(y16,4))

        #bond7
        p17 = float(bond7[date[i]])
        coupon7 = float(bond7['Coupon'])
        y17 = ytm(p17,Daycount1[i],coupon7,7)
        ytm_list[i].append(round(y17,4))

        #bond8
        p18 = float(bond8[date[i]])
        coupon8 = float(bond8['Coupon'])
        y18 = ytm(p18,Daycount2[i],coupon8,7)
        ytm_list[i].append(round(y18,4))

        #bond9
        p19 = float(bond9[date[i]])
        coupon9 = float(bond9['Coupon'])
        y19 = ytm(p19,Daycount1[i],coupon9,9)
        ytm_list[i].append(round(y19,4))

        #bond10
        p110 = float(bond10[date[i]])
        coupon10 = float(bond10['Coupon'])
        y110 = ytm(p110,Daycount1[i],coupon10,10)
        ytm_list[i].append(round(y110,4))

        #bond11
        p111 = float(bond11[date[i]])
        coupon11 = float(bond11['Coupon'])
        y111 = ytm1(p111,Daycount1[i],coupon11,11)
        ytm_list[i].append(round(y111,4))

    return ytm_list


def spotrate(pv,fv,t):
    spot = 0.0000001
    while abs(pv*((1+spot/2)**(t/182.5)) - fv) >= 0.0001:
        spot += 0.0000001

    return spot

def spotrateforbond7and9(pv,fv,t,lastspot,coupon):  # for bond7 and bond9 only. Need to linear interpolate the spot rate
                                                    # used for coupon on 9/1. 
    spot = 0.0000001
    spot_for_last_coupon = round(np.interp(92,[0,273],[lastspot,spot]),7)
    while fv - ((pv-(coupon/2)/((1+spot_for_last_coupon/2)**((t+92)/182.5))) * ((1+spot/2)**((t+273)/182.5)))>= 0.0001:
        spot += 0.0000001
        spot_for_last_coupon = np.interp(92, [0, 273], [lastspot, spot])
    return spot


def spot_curve(bond1,bond2,bond3,bond4,bond5,bond6,bond7,bond8,bond9,bond10,bond11,date,T):
    spot_list = [[],[],[],[],[],[],[],[],[],[]]
    Daycount1 = [59,58,55,54,53,52,51,48,47,46]
    Daycount2 = [151, 150, 147, 146, 145, 144, 143, 140, 139, 138]

    for i in range(len(date)):
        #bond 1
        p11 = float(bond1[date[0]])
        coupon1 = float(bond1['Coupon'])*100
        fv1 = 100 + coupon1/2
        pv1 = (p11 + coupon1  * (182 - Daycount1[i]) / 365)
        spot_1 = spotrate(pv1,fv1,Daycount1[i])
        spot_list[i].append(round(spot_1,4))

        #bond 2
        p12 = float(bond2[date[i]])
        coupon2 = float(bond2['Coupon'])*100
        fv2 = 100 + coupon2/2
        pv2 = (p12 + coupon2 * (182 - Daycount1[i]) / 365) - (coupon2/2)/((1+spot_1/2)**(Daycount1[i]/182.5))
        spot_2 = spotrate(pv2,fv2,Daycount1[i]+184)
        spot_list[i].append(round(spot_2,4))

        #bond 3
        p13 = float(bond3[date[i]])
        coupon3 = float(bond3['Coupon'])*100
        fv3 = 100 + coupon3/2
        pv3 = (p13 + coupon3  * (182 - Daycount1[i]) / 365) - (coupon3/2)/((1+spot_1/2)**(Daycount1[i]/182.5)) - (coupon3/2)/((1+spot_2/2)**((Daycount1[i]+184)/182.5))
        spot_3 = spotrate(pv3, fv3, Daycount1[i] + 184 + 181)
        spot_list[i].append(round(spot_3,4))

        #bond 4
        p14 = float(bond4[date[i]])
        coupon4 = float(bond4['Coupon'])*100
        fv4 = 100 + coupon4/2
        pv4 = (p14 + coupon4 * (182 - Daycount1[i]) / 365) - (coupon4/2)/((1+spot_1/2)**(Daycount1[i]/182.5)) - (coupon4/2)/((1+spot_2/2)**((Daycount1[i]+184)/182.5)) - \
             (coupon4/2)/((1+spot_3/2)**((Daycount1[i]+184+181)/182.5))
        spot_4 = spotrate(pv4, fv4, Daycount1[i] + 184 + 181 + 184)
        spot_list[i].append(round(spot_4,4))

        #bond 5
        p15 = float(bond5[date[i]])
        coupon5 = float(bond5['Coupon'])*100
        fv5 = 100 + coupon5/2
        pv5 = (p15 + coupon5  * (182 - Daycount1[i]) / 365) - (coupon5/2)/((1+spot_1/2)**(Daycount1[i]/182.5)) - (coupon5/2)/((1+spot_2/2)**((Daycount1[i]+184)/182.5)) - \
             (coupon5/2)/((1+spot_3/2)**((Daycount1[i]+184+181)/182.5)) - (coupon5/2)/((1+spot_4/2)**((Daycount1[i]+184+181+184)/182.5))
        spot_5 = spotrate(pv5, fv5, Daycount1[i] + 184 + 181 + 184 + 181)
        spot_list[i].append(round(spot_5,4))


        #bond 6
        spot_for_bond6 = []
        for j in range(4):
            inter = np.interp(Daycount2[i]/365+j*0.5, T[i][0:5], spot_list[i])
            spot_for_bond6.append(inter)


        p16 = float(bond6[date[i]])
        coupon6 = float(bond6['Coupon'])*100
        fv6 = 100 + coupon6/2
        pv6 = (p16 + coupon6  * (183 - Daycount2[i]) / 365) - (coupon6/2)/((1+spot_for_bond6[0]/2)**(Daycount2[i]/182.5)) - (coupon6/2)/((1+spot_for_bond6[1]/2)**((Daycount2[i]+183)/182.5)) - \
             (coupon6/2)/((1+spot_for_bond6[2]/2)**((Daycount2[i]+183+182)/182.5)) - (coupon6/2)/((1+spot_for_bond6[3]/2)**((Daycount2[i]+183+182+183)/182.5))
        spot_6 = spotrate(pv6, fv6, Daycount2[i] + 183 + 182 + 183 + 182)  # 22, 6/1
        spot_list[i].append(round(spot_6,4))


        #bond 7
        p17 = float(bond7[date[i]])
        coupon7 = float(bond7['Coupon'])*100
        fv7 = 100 + coupon7/2
        pv7 = (p17 + coupon7 * (182 - Daycount1[i]) / 365) - (coupon7/2)/((1+spot_1/2)**(Daycount1[i]/182.5)) - (coupon7/2)/((1+spot_2/2)**((Daycount1[i]+184)/182.5)) - \
             (coupon7/2)/((1+spot_3/2)**((Daycount1[i]+184+181)/182.5)) - (coupon7/2)/((1+spot_4/2)**((Daycount1[i]+184+181+184)/182.5)) - \
             (coupon7/2)/((1+spot_5/2)**((Daycount1[i]+184+181+184+181) / 182.5))
        spot_7 = spotrateforbond7and9(pv7, fv7, Daycount1[i] + 184 + 181 + 184 + 181 + 92,spot_6,coupon7)
        spot_list[i].append(round(spot_7,4))


        #bond 8
        spot_for_bond8 = []
        for j in range(6):
            inter = np.interp(Daycount2[i]/365+j*0.5, T[i][0:7], spot_list[i])
            spot_for_bond8.append(inter)

        p18 = float(bond8[date[i]])
        coupon8 = float(bond8['Coupon'])*100
        fv8 = 100 + coupon8/2
        pv8 = (p18 + coupon8  * (182 - Daycount2[i]) / 365) - (coupon8/2)/((1+spot_for_bond8[0]/2)**(Daycount2[i]/182.5)) - (coupon8/2)/((1+spot_for_bond8[1]/2)**((Daycount2[i]+183)/182.5)) - \
             (coupon8/2)/((1+spot_for_bond8[2]/2)**((Daycount2[i]+183+182)/182.5)) - (coupon8/2)/((1+spot_for_bond8[3]/2)**((Daycount2[i]+183+182+183)/182.5)) - \
             (coupon8/2)/((1+spot_for_bond8[4]/2)**((Daycount2[i]+183+182+183+182) / 182.5)) - (coupon8/2)/((1+spot_for_bond8[5]/2)**((Daycount2[i]+183+182+183+182+183) / 182.5))
        spot_8 = spotrate(pv8, fv8, Daycount2[i] + 183 + 182 + 183 + 182 + 183 + 182)
        spot_list[i].append(round(spot_8,4))

        #bond 9
        spot_sixcoupon = np.interp(Daycount1[i]+ 0.5*5,T[i][0:8],spot_list[i])
        p19 = float(bond9[date[i]])
        coupon9 = float(bond9['Coupon'])*100
        fv9 = 100 + coupon9/2
        pv9 = (p19 + coupon9 * (182 - Daycount1[i]) / 365) - (coupon9/2)/((1+spot_1/2)**(Daycount1[i]/182.5)) - (coupon9/2)/((1+spot_2/2)**((Daycount1[i]+184)/182.5)) - \
             (coupon9/2)/((1+spot_3/2)**((Daycount1[i]+184+181)/182.5)) - (coupon9/2)/((1+spot_4/2)**((Daycount1[i]+184+181+184)/182.5)) - \
             (coupon9/2)/((1+spot_5/2)**((Daycount1[i]+184+181+184+181) / 182.5)) - (coupon9/2)/((1+spot_sixcoupon/2)**((Daycount1[i]+184+181+184+181+92) / 182.5)) - \
              (coupon9 / 2) / ((1 + spot_7 / 2) ** ((Daycount1[i] + 184 + 181 + 184 + 181 + 92 + 273) / 182.5))
        spot_9 = spotrateforbond7and9(pv9, fv9, Daycount1[i] + 184 + 181 + 184 + 181 + 92 + 273 + 92, spot_8,coupon9)
        spot_list[i].append(round(spot_9,4))

        #bond 10
        spot_eightcoupon = np.interp(Daycount1[i]+ 0.5*7,T[i][0:9],spot_list[i])
        p110 = float(bond10[date[i]])
        coupon10 = float(bond10['Coupon'])*100
        fv10 = 100 + coupon10/2
        pv10 = (p110 + coupon10 * (182 - Daycount1[i]) / 365) - (coupon10/2)/((1+spot_1/2)**(Daycount1[i]/182.5)) - (coupon10/2)/((1+spot_2/2)**((Daycount1[i]+184)/182.5)) - \
             (coupon10/2)/((1+spot_3/2)**((Daycount1[i]+184+181)/182.5)) - (coupon10/2)/((1+spot_4/2)**((Daycount1[i]+184+181+184)/182.5)) - \
             (coupon10/2)/((1+spot_5/2)**((Daycount1[i]+184+181+184+181) / 182.5)) - (coupon10/2)/((1+spot_sixcoupon/2)**((Daycount1[i]+184+181+184+181+92) / 182.5)) - \
              (coupon10 / 2) / ((1 + spot_7 / 2) ** ((Daycount1[i] + 184 + 181 + 184 + 181 + 92 + 273) / 182.5)) - (coupon10/2)/((1+spot_eightcoupon/2)**((Daycount1[i]+184+181+184+181+92+273+92) / 182.5)) - \
               (coupon10 / 2) / ((1 + spot_9 / 2) ** ((Daycount1[i] + 184 + 181 + 184 + 181 + 92 + 273 + 92 + 274) / 182.5))
        spot_10 = spotrate(pv10, fv10, Daycount1[i] + 184 + 181 + 184 + 181 + 92 + 273 + 92+274+184)
        spot_list[i].append(round(spot_10,4))

        #bond 11
        p111 = float(bond11[date[i]])
        coupon11 = float(bond11['Coupon'])*100
        fv11 = 100 + coupon11/2
        pv11 = (p111 + coupon11 * (141 - Daycount1[i]) / 365) - (coupon11/2)/((1+spot_1/2)**(Daycount1[i]/182.5)) - (coupon11/2)/((1+spot_2/2)**((Daycount1[i]+184)/182.5)) - \
             (coupon11/2)/((1+spot_3/2)**((Daycount1[i]+184+181)/182.5)) - (coupon11/2)/((1+spot_4/2)**((Daycount1[i]+184+181+184)/182.5)) - \
             (coupon11/2)/((1+spot_5/2)**((Daycount1[i]+184+181+184+181) / 182.5)) - (coupon11/2)/((1+spot_sixcoupon/2)**((Daycount1[i]+184+181+184+181+92) / 182.5)) - \
              (coupon11 / 2) / ((1 + spot_7 / 2) ** ((Daycount1[i] + 184 + 181 + 184 + 181 + 92 + 273) / 182.5)) - (coupon11/2)/((1+spot_eightcoupon/2)**((Daycount1[i]+184+181+184+181+92+273+92) / 182.5)) - \
               (coupon11 / 2) / ((1 + spot_9 / 2) ** ((Daycount1[i]+184+ 81+184+181+92+ 73+92+274) / 182.5)) - (coupon11/2)/((1+spot_10/2)**((Daycount1[i]+184+181+184+181+92+273+92+274+184) / 182.5))
        spot_11 = spotrate(pv11, fv11, Daycount1[i] + 184 + 181 + 184 + 181 + 92 + 273 + 92+274+184+181)
        spot_list[i].append(round(spot_11,4))

    return spot_list







if __name__ == '__main__':
    data = pd.read_excel(r'~/Downloads/APM466/Assignment1/Bond.xlsx')
    bond1 = data.loc[data['ISIN'] == 'CA135087D929']
    bond2 = data.loc[data['ISIN'] == 'CA135087E596']
    bond3 = data.loc[data['ISIN'] == 'CA135087F254']
    bond4 = data.loc[data['ISIN'] == 'CA135087F585']
    bond5 = data.loc[data['ISIN'] == 'CA135087G328']
    bond6 = data.loc[data['ISIN'] == 'CA135087ZU15']
    bond7 = data.loc[data['ISIN'] == 'CA135087H490']
    bond8 = data.loc[data['ISIN'] == 'CA135087A610']
    bond9 = data.loc[data['ISIN'] == 'CA135087J546']
    bond10 = data.loc[data['ISIN'] == 'CA135087J967']
    bond11 = data.loc[data['ISIN'] == 'CA135087K528']
    date = ['1/2/2020','1/3/2020','1/6/2020','1/7/2020','1/8/2020','1/9/2020','1/10/2020','1/13/2020','1/14/2020','1/15/2020']
    Daycount = [59, 58, 55, 54, 53, 52, 51, 48, 47, 46]  # num of days between settlement date and 3/1/2020
    T = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(10):
        for j in range(11):
            if j == 0:
                T[i].append(Daycount[i] / 365)
            elif j==5 or j == 7:
                T[i].append(T[i][j - 1] + 0.25)
            elif j==6 or j==8:
                T[i].append(T[i][j - 1] + 0.75)
            else:
                T[i].append(T[i][j-1] + 0.5)

    legend = ['Jan.2', 'Jan.3', 'Jan.6', 'Jan.7', 'Jan.8', 'Jan.9', 'Jan.10', 'Jan.13', 'Jan.14', 'Jan.15']
    # yield curve
    fi, fg = plt.subplots(3)
    ytm_list = ytm_for_eachday(bond1, bond2, bond3, bond4, bond5, bond6, bond7, bond8, bond9, bond10, bond11, date)
    for i in range(10):
        fg[0].plot(T[i],ytm_list[i],label = legend[i])
        fg[0].legend(loc='upper right',bbox_to_anchor=(1.1,1.5))

    fg[0].set_title('Yield Curve')
    fg[0].set_xlabel('Time')
    fg[0].set_ylabel('Interest Rate')

    # get yield for 1-5 year
    yield_1to5 = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(10):
        for j in range(5):
            inter_yield= np.interp(j+1, T[i],ytm_list[i])
            yield_1to5[i].append(round(inter_yield,4))
    print(yield_1to5)

    # spot curve
    spot_list = spot_curve(bond1, bond2, bond3, bond4, bond5, bond6, bond7, bond8, bond9, bond10, bond11, date,T)
    for i in range(10):
        fg[1].plot(T[i], spot_list[i],label = legend[i])
        fg[1].legend(loc='upper right', bbox_to_anchor=(1.1, 1.3))

    fg[1].set_title('Spot Curve')
    fg[1].set_xlabel('Time')
    fg[1].set_ylabel('Interest Rate')

    # get 1-5 yr spot rate
    spot_1to5 = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(10):
        for j in range(5):
            inter = np.interp(j+1, T[i],spot_list[i])
            spot_1to5[i].append(round(inter,4))
    print(spot_1to5)

    # forward curve
    forward_list = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(10):
        for j in range(4):
            forward_rate = (((1+spot_1to5[i][j+1])**(j+2))/(1+spot_1to5[i][0]))**(1/(j+1)) - 1
            forward_list[i].append(round(forward_rate,4))
    print(forward_list)

    for i in range(10):
        fg[2].plot(['1yr-1yr','1yr-2yr','1yr-3yr','1yr-4yr'],forward_list[i],label = legend[i])
        fg[2].legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    fg[2].set_title('Forward Curve')
    fg[2].set_xlabel('Time')
    fg[2].set_ylabel('Interest Rate')
    plt.show()

    # calculate covariance matrix for log-return of yield & forward 
    ret_list = [[],[],[],[],[]]
    for i in range(5):
        for j in range(9):
            ret = math.log(yield_1to5[j+1][i]/yield_1to5[j][i])
            ret_list[i].append(ret)
    x = np.array(ret_list)
    cov1 = np.cov(x)

    forward_array = [[],[],[],[]]
    for i in range(4):
        for j in range(9):
            forward_array[i].append(math.log(forward_list[j+1][i]/forward_list[j][i]))
    y = np.array(forward_array)
    cov2 = np.cov(y)
    print(cov1)
    print(cov2)

    # calculate eigenvalues and eigenvectors. 
    eigval1,eigvec1 = np.linalg.eig(cov1)
    eigval2,eigvec2 = np.linalg.eig(cov2)
    print(eigval1)
    print(eigvec1)
    print(eigval2)
    print(eigvec2)








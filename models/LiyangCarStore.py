#-*- coding: UTF-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import pandas as pd


class LiyangCarStore(object):
    numerical_cols = 'price_guide_min price_guide_max xinglixiangrongji_min xinglixiangrongji_max gongxinbuzongheyouhao pailiang chemenshu dangweishu zuixiaozhuanwanbanjing tingchannianfen zuoweishu zuigaochesu zuidazaizhongzhiliang chang kuan gao qianlunju houlunju zuidagonglv zuidaniuju zuidamali zuidaniujuzhuansu zuidagonglvzhuansu qigangrongji youxiangrongji zhengbeizhiliang zuixiaolidijianxi yangshengqishuliang qigangshu niankuan shengchannianfen shangshinianfen jiasushijian zhouju yasuobi meigangqimenshu shangshiyuefen'.split()
    ordinal_cols = 'xuniduodieCD anquandaiweixitishi LATCHzuoyijiekou houpaisandianshianquandai anquandaiyushoujingongneng zheyangbanhuazhuangjing zidongzhuche_shangpofuzhu fujiashianquanqinang zidongtoudeng zuoyitongfeng zhenpizuoyi zhenpifangxiangpan zhongkongtaicaisedaping houzuochufengkou kongqidiaojie_huafenguolv bochefuzhu houpaiceqinang fangxiangpanhuandang qianpaitoubuqinang houzuozhongyangfushou houpaizuoyidiandongdiaojie houshijingzidongfangxuanmu duogongnengfangxiangpan jiashizuozuoyidiandongdiaojie dingweihudongfuwu houpaibeijia zidongkongdiao zhuanxiangtoudeng xibuqinang fangxiangpandiandongdiaojie zishiyingxunhang rijianxingchedeng xianqidadeng qianpaiceqinang duodieDVD chezaixinxifuwu fujiashizuozuoyidiandongdiaojie chezaidianshi houpaidulikongdiao diandongxihemen wuyaoshiqidongxitong jianbuzhichengdiaojie zhongkongyejingpingfenpingxianshi dierpaizuoyiyidong houdiandongchechuang yeshixitong cheneifenweideng dierpaikaobeijiaodudiaojie houshijingjiyi zhudongchache qianwudeng wendufenqukongzhi xingchediannaoxianshiping houshijingjiare cheshenwendingkongzhi dingsuxunhang chechuangfangjiashougongneng daocheshipinyingxiang LEDdadeng qianyinlikongzhi zhudongzhuanxiangxitong jiashizuoanquanqinang ABSfangbaosi houyushua houpaizuoyibilifangdao daocheleida yaokongyaoshi houpaiyejingping lanya/chezaidianhua quanjingtianchuang qiandiandongchechuang bingxianfuzhu disanpaizuoyi fadongjidianzifangdao neizhiyingpan dadenggaodukediao quanjingshexiangtou zhongkongsuo kebianxuangua qianzuozhongyangfushou houshijingdiandongdiaojie chachefuzhu doupohuanjiang diandongtianchuang fangxiangpanqianhoudiaojie taiyajiancezhuangzhi ganyingyushua gereboli dadengqingxizhuangzhi dandieCD fangxiangpanshangxiadiaojie qianpaizuoyijiare GPSdaohang zidongbocheruwei diandongzuoyijiyi zhidonglifenpei houpaicezheyanglian lingtaiyajixuxingshi yundongzuoyi houpaizuoyijiare zuoyianmo yinpinzhichiMP3 HUDtaitoushuzixianshi kongdiao houpaizuoyizhengtifangdao houpaitoubuqinang renjijiaohuxitong houshijingdiandongzhedie kebianzhuanxiangbi chezaibingxiang waijieyinyuanjiekou yaobuzhichengdiaojie ISOFIXertongzuoyijiekou kongqixuangua zuoyigaodidiaojie dandieDVD duodieCD diandonghoubeixiang houfengdangzheyanglian yundongwaiguantaojian'.split()
    categorical_cols = 'cheshenyanse qianlunguguige houlunguguige ranyoubiaohao biansuxiangleixing chetijiegou cheliangjibie cheliangleixing cheshenxingshi fadongjiweizhi ranyouleixing qudongxingshi qudongfangshi gongyoufangshi jinqixingshi qigangpailiexingshi zhuanxiangjixingshi shengchanzhuangtai guochanhezijinkou guobie pinpai changjia tag_id chexi biansuqimiaoshu paifangbiaozhun zhulileixing lungucailiao qianzhidongqileixing houzhidongqileixing qianxuangualeixing houxuangualeixing'.split()

    def __init__(self, file_data, file_desc=None):
        self.data = pd.read_csv(file_data, sep='\t', quoting=3, dtype=str).set_index('id')
        self.desc = None if file_desc is None else pd.read_csv(file_desc, sep='\t', quoting=3, names=['col', 'desc'])

    def get_data(self):
        reto = self.get_ordinals()
        retn = self.get_numericals()
        retc = self.get_categoricals()
        #print retn
        ret = pd.concat([retn, reto, retc], axis=1)
        ret.index = ret.index.map(int)
        return ret, retn.columns.tolist() + reto.columns.tolist(), retc.columns.tolist()

    def get_numericals(self):
        df = self.data
        df['pailiang'] = df.pailiang.replace('电动', '-1').astype(float)  # TEMP
        df['price_guide_min'] = df.zhidaojiage.fillna('0').replace('待查', '0').str.split(r'[~-]').str[0].astype(float) * 10000
        #print df['price_guide_min']
        df['price_guide_max'] = df.zhidaojiage.fillna('0').replace('待查', '0').str.split(r'[~-]').str[-1].astype(float) * 10000
        df['xinglixiangrongji_min'] = df.xinglixiangrongji.fillna('0').str.split(r'\D+').str[0].replace('', '0').astype(int)
        df['xinglixiangrongji_max'] = ((df.xinglixiangrongji.fillna('0') + '-') * 2).str.split(r'\D+').str[1].replace('', '0').astype(int)
        df['gongxinbuzongheyouhao'] = df.gongxinbuzongheyouhao.fillna('0').replace('9月13日', '0').str.split(r'[-/(]').str[0].astype(float)
        df['chemenshu'] = df.chemenshu.map({'两门': 2, '三门': 3, '四门': 4, '五门': 5, '六门': 6})
        df['dangweishu'] = df.dangweishu.fillna('0').replace('无级', '10').str.extract(r'(\d+)', expand=False).astype(int)
        df['zuixiaozhuanwanbanjing'] = df.zuixiaozhuanwanbanjing.fillna('0').str.split('/').str[0].astype(float)
        df['tingchannianfen'] = df.tingchannianfen.replace(u'\u2014'.encode('utf8'), 0).astype(int)
        for col in ['zuoweishu', 'zuigaochesu', 'zuidazaizhongzhiliang', 'chang', 'kuan', 'gao', 'qianlunju', 'houlunju', 'zuidagonglv', 'zuidaniuju', 'zuidamali', 'zuidaniujuzhuansu', 'zuidagonglvzhuansu', 'qigangrongji', 'youxiangrongji', 'zhengbeizhiliang', 'zuixiaolidijianxi', 'yangshengqishuliang']:
            df[col] = df[col].fillna('0').astype(str).str.extract(r'(\d+)', expand=False).fillna('0').astype(int)
        for col in ['qigangshu', 'niankuan', 'shengchannianfen', 'shangshinianfen', 'meigangqimenshu', 'shangshiyuefen']:
            df[col] = df[col].fillna('0').astype(int)
        for col in ['jiasushijian', 'yasuobi', 'zhouju']:
            df[col] = df[col].fillna('0').astype(float)
        #print df[self.numerical_cols]
        return df[self.numerical_cols]

    def get_ordinals(self):
        #print self.ordinal_cols
        for col in self.ordinal_cols:
            self.data[col] = self.data[col].map(
                {'无': 1, '无无': 1, '选配': 2, '选装': 2, '有?': 3, '有': 4, 'USB+AUX': 4}).fillna(0)
        #print self.data[self.ordinal_cols]
        return self.data[self.ordinal_cols]

    def get_categoricals(self):
        for col in ['qianxuangualeixing', 'houxuangualeixing']:
            self.data[col] = self.data[col].apply(
                lambda s: s.decode('utf8')[:3].encode('utf8') if isinstance(s, str) else s)
        return self.data[self.categorical_cols]

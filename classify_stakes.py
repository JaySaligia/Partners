import os
import pandas as pd
from tag_process.tagprocessstakes import IndClassifyStake
from tqdm import trange
from multiprocessing import Pool


class IndClassify_stakes:
    def __init__(self, data_dir, corp_cuts, patent_cuts, adds_cuts, tag_dir, res_dir):
        # 企业文件目录
        self.corp_data_adr = data_dir
        # 企业分隔路径
        self.corp_cuts_adr = os.path.join(self.corp_data_adr, corp_cuts)
        # 专利分隔路径
        self.patent_cuts_adr = os.path.join(self.corp_data_adr, patent_cuts)
        # 存储部分目录
        self.part_adds_adr = os.path.join(self.corp_data_adr, adds_cuts)
        # 相关需要文件目录
        self.corp_infor_adr = os.path.join(self.corp_data_adr, 'corp_inforn.csv')
        # 初始化分类
        # self.T = IndClassifyStake(244, 88, tag_dir)
        self.T = IndClassifyStake(244, 1381, tag_dir)
        # self.T = 0
        # 分类结果目录
        self.res_adr = res_dir
        # 并行数
        self.process_n = 5

    def classify(self):
        # 列出目录下对应excel文件
        chunksize = 100000
        excel_list = os.listdir(self.corp_cuts_adr)
        # # 文件按顺序输出
        name_list = sorted(excel_list, key=lambda x: int(x.replace("query_result", "").split('.')[0]))
        # # 文件路径遍历
        excel_dirs = [os.path.join(self.corp_cuts_adr, i) for i in name_list]
        # # 第1列数据为索引
        pds_list = [pd.read_excel(i, header=0, index_col=0, encoding="utf-8", dtype={'corp_infor.corp_id': str}) for i
                    in excel_dirs]
        # # add文件横向合并
        corp_infor = pd.concat(pds_list, axis=0)
        # # 替换\t字符，为空字符串
        corp_infor = corp_infor.replace('\t+', '', regex=True)
        # # 保存csv格式,以\t分隔
        corp_infor.to_csv(self.corp_infor_adr, encoding="utf-8", sep='\t')
        corp_raw = pd.read_csv(self.corp_infor_adr, header=0, encoding="utf-8", sep='\t',
                               dtype={'corp_infor.corp_id': str})
        corp_num = len(corp_raw)
        corp_infor = pd.read_csv(self.corp_infor_adr, header=0, encoding="utf-8", sep='\t', chunksize=chunksize,
                                 dtype={'corp_infor.corp_id': str})
        print('------当前并行数:%d------' % self.process_n)
        print('--------总计%d家企业待处理-------' % corp_num)
        # 分块处理
        chunkall = corp_num // chunksize + 1
        print('--------总计分为%d部分处理-------' % chunkall)
        chunkcount = 0
        for chunk in corp_infor:
            chunkcount += 1
            print('--------正在计算区块{}/{}-------'.format(chunkcount, chunkall))
            self.indclassify(chunk, chunkcount)
        print('--------分块计算完毕-------')
        # 列出目录下对应csv文件
        csv_list = os.listdir(self.part_adds_adr)
        # 文件按顺序输出
        csv_list = sorted(csv_list, key=lambda x: int(x.replace("query_result", "").split('.')[0]))
        csv_dirs = [os.path.join(self.part_adds_adr, i) for i in csv_list]
        # 拼接结果
        add_pds = [pd.read_csv(i, header=0, index_col=0, encoding='utf-8', sep='\t', dtype={'corp_infor.corp_id': str})
                   for i in csv_dirs]
        adds_part = pd.concat(add_pds, axis=0, ignore_index=False)
        # join操作
        corp_out = pd.merge(corp_raw, adds_part, how='inner', on='corp_infor.corp_id')
        print(corp_out)
        corp_out.to_csv(self.res_adr, encoding='utf-8', sep='\t')
        print('--------产业分类计算完毕-------')

    # 多进程处理分区
    def zoomlabel(self, chunk, chunkcount):
        res = {}
        chunk.fillna(value='-1000', inplace=True)
        # 多进程运算对每个chunk结果进行分类并保存结果
        # 将chunk根据进程数拆分
        slicesize = len(chunk) // self.process_n
        chunklist = [chunk.iloc[i * slicesize:(i + 1) * slicesize, :] for i in range(self.process_n - 1)]
        chunklist.append(chunk.iloc[(self.process_n - 1) * slicesize:, :])
        pool = Pool(processes=self.process_n)
        results = []
        for i in range(self.process_n):
            results.append(pool.apply_async(self.multifunc_zoom, args=(chunklist[i], i)))
        pool.close()
        pool.join()
        # 结果
        res_ids = []
        res_zooms = []
        for result in results:
            item = result.get()
            res_ids.extend(item[0])
            res_zooms.extend(item[1])
        # 列出结果
        res['corp_infor.corp_id'] = res_ids
        res['corp_infor.zoom'] = res_zooms
        df = pd.DataFrame(res)
        df.to_csv(os.path.join(self.part_adds_adr, 'query_result%d.csv' % chunkcount), encoding='utf-8', sep='\t')

    def multifunc_zoom(self, chunkslice, processindex):
        ids = []
        zooms = []
        for i in trange(len(chunkslice)):
            row = chunkslice.iloc[i]
            id = row['corp_infor.corp_id']
            address = row['corp_infor.address']
            authority = row['corp_infor.authority']
            zoomres = self.processzoom(address, authority)
            ids.append(id)
            zooms.append(zoomres)
        return [ids, zooms]

    def judge(self, name):
        if '武汉东湖' in name or '武汉市东湖' in name or '东湖开发区' in name:
            new = '武汉东湖新技术开发区'
        elif '武汉临空港' in name or '武汉市临空港' in name or '临空港开发区' in name:
            new = '武汉临空港经济技术开发区'
        elif '武汉经济' in name or '武汉市经济' in name:
            new = '武汉经济技术开发区'
        else:
            new = '武汉非经济开发区'
        return new

    def processzoom(self, corp_address, corp_authority):
        if self.judge(corp_address) != '武汉非经济开发区' or self.judge(corp_authority) != '武汉非经济开发区':
            # 根据企业地址和工商局信息
            if self.judge(corp_address) != '武汉非经济开发区':
                # 两者取其一
                new = self.judge(corp_address)
            else:
                new = self.judge(corp_authority)
        else:
            new = '武汉非经济开发区'
        return new

    # 多进程处理企业分类
    def indclassify(self, chunk, chunkcount):
        res = {}
        chunk.fillna(value='-1000', inplace=True)
        # 多进程运算对每个chunk结果进行分类并保存结果
        # 将chunk根据进程数拆分
        slicesize = len(chunk) // self.process_n
        chunklist = [chunk.iloc[i * slicesize:(i + 1) * slicesize, :] for i in range(self.process_n - 1)]
        chunklist.append(chunk.iloc[(self.process_n - 1) * slicesize:, :])
        pool = Pool(processes=self.process_n)
        results = []
        for i in range(self.process_n):
            results.append(pool.apply(self.multifunc, args=(chunklist[i], i)))
        pool.close()
        pool.join()
        # 结果
        res_ids = []
        res_ind1 = []
        res_ind2 = []
        res_ind3 = []
        for result in results:
            item = result
            res_ids.extend(item[0])
            res_ind1.extend(item[1])
            res_ind2.extend(item[2])
            res_ind3.extend(item[3])
        # 列出结果
        res['corp_infor.corp_id'] = res_ids
        res['corp_infor.indus1'] = res_ind1
        res['corp_infor.indus2'] = res_ind2
        res['corp_infor.indus3'] = res_ind3
        df = pd.DataFrame(res)
        df.to_csv(os.path.join(self.part_adds_adr, 'query_result%d.csv' % chunkcount), encoding='utf-8', sep='\t')

    def multifunc(self, chunkslice, processindex):
        ids = []
        ind_1 = []
        ind_2 = []
        ind_3 = []
        for i in trange(len(chunkslice)):
            row = chunkslice.iloc[i]
            id = row['corp_infor.corp_id']
            scope = row['corp_infor.business_scope']
            tags = row['corp_infor.tags']
            classifyres = self.processclassify(scope, tags)
            ids.append(id)
            ind_1.append(classifyres[0])
            ind_2.append(classifyres[1])
            ind_3.append(classifyres[2])
        return [ids, ind_1, ind_2, ind_3]

    # 处理经营范围和标签
    def processclassify(self, scope, tags):
        # 计算出企业分类
        top_k = 3
        corp_tags = tags
        corp_terms = scope
        ids = self.T.single_match_scope(corp_terms, corp_tags)
        nids = ids[:top_k] if len(ids) > top_k else ids + ['未知行业'] * (top_k - len(ids))
        return nids


if __name__ == '__main__':
    corp = IndClassify_stakes('./corp_data/', 'corp_inforns/', 'corp_patents/', 'corp_addeds/', './tag_process/',
                              './result/wuhan_1381.csv')
    corp.classify()

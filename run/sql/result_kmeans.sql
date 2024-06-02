CREATE TABLE result_kmeans
(
    id                           bigint       NOT NULL AUTO_INCREMENT COMMENT '主键',
    dataset_name                 varchar(255) NOT NULL DEFAULT '' COMMENT '数据集名称',
    instance                     int          NOT NULL DEFAULT 0 COMMENT '记录数',
    run_id                       varchar(64)  NOT NULL DEFAULT '' COMMENT '运行id',
    batch_run_id                 varchar(64)  NOT NULL DEFAULT '' COMMENT '批量运行id',
    create_time                  timestamp    NOT NULL DEFAULT current_timestamp COMMENT '创建时间',
    k                            int          NOT NULL DEFAULT 0 COMMENT 'k',
    tag                          varchar(10)  NOT NULL DEFAULT '' COMMENT '标记',
    seed                         int(10) NOT NULL DEFAULT 0 COMMENT '随机种子',
    homogeneity_score            double       NOT NULL DEFAULT 0 COMMENT 'k',
    completeness_score           double       NOT NULL DEFAULT 0 COMMENT 'k',
    v_measure_score              double       NOT NULL DEFAULT 0 COMMENT 'k',
    adjusted_rand_score          double       NOT NULL DEFAULT 0 COMMENT 'k',
    adjusted_mutual_info_score   double       NOT NULL DEFAULT 0 COMMENT 'k',
    fowlkes_mallows_score        double       NOT NULL DEFAULT 0 COMMENT 'k',
    rand_score                   double       NOT NULL DEFAULT 0 COMMENT 'k',
    mutual_info_score            double       NOT NULL DEFAULT 0 COMMENT 'k',
    normalized_mutual_info_score double       NOT NULL DEFAULT 0 COMMENT 'k',
    PRIMARY KEY (id)
) ENGINE=innoDB DEFAULT CHARSET=utf8 comment 'kmeans运行结果保存';

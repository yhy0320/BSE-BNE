CREATE TABLE result_dt
(
    id           bigint       NOT NULL AUTO_INCREMENT COMMENT '主键',
    dataset_name varchar(255) NOT NULL DEFAULT '' COMMENT '数据集名称',
    instance     int          NOT NULL DEFAULT 0 COMMENT '记录数',
    run_id       varchar(64)  NOT NULL DEFAULT '' COMMENT '运行id',
    batch_run_id varchar(64)  NOT NULL DEFAULT '' COMMENT '批量运行id',
    create_time  timestamp    NOT NULL DEFAULT current_timestamp COMMENT '创建时间',
    max_depth    int          NOT NULL DEFAULT 0 COMMENT 'max_depth',
    tag          varchar(10)  NOT NULL DEFAULT '' COMMENT '标记',
    seed         int(10) NOT NULL DEFAULT 0 COMMENT '随机种子',
    score        double       NOT NULL DEFAULT 0 COMMENT 'k',
    PRIMARY KEY (id)
) ENGINE=innoDB DEFAULT CHARSET=utf8 comment 'dt运行结果保存';

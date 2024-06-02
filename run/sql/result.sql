CREATE TABLE result
(
    id           bigint        NOT NULL AUTO_INCREMENT COMMENT '主键',
    dataset_name varchar(255)  NOT NULL DEFAULT '' COMMENT '数据集名称',
    instance     int           NOT NULL DEFAULT 0  COMMENT '数据集记录数',
    dimension    int           NOT NULL DEFAULT 0  COMMENT '数据集维度',
    cost_time    varchar(64)   NOT NULL DEFAULT '' COMMENT '消耗时间',
    params       varchar(1024) NOT NULL DEFAULT '' COMMENT '运行参数',
    metrics      varchar(1024) NOT NULL DEFAULT '' COMMENT '结果指标',
    file_path    varchar(256)  NOT NULL DEFAULT '' COMMENT '生成结果文件路径',
    create_time  timestamp     NOT NULL DEFAULT current_timestamp COMMENT '创建时间',
    run_id       varchar(64)   NOT NULL DEFAULT '' COMMENT '运行id',
    tag          varchar(10)   NOT NULL DEFAULT '' COMMENT '标记',
    PRIMARY KEY (id)
) ENGINE=innoDB DEFAULT CHARSET=utf8 comment '运行结果';
